"""Sequential deletion stream with sharded exact unlearning vs a DynamicFU re-implementation (exploratory, not preregistered; results in results/shard_exploration).

Fixed before running:
- data connected_v3_seed42; policies may read VALIDATION (the FU aux set); reported metrics on CALIBRATION; test unused;
- 20 clients, Dirichlet alpha 0.5, partition seed 42; shards S in {4, 10} by seeded permutation (seed 7); shard j seed s+1000j;
- FedAvg per shard: 15 rounds (R_base), 1 local epoch, Adam 1e-3, batch 512; ensemble = sum of shard margins;
- training seeds 10042..10046;
- stream: 10 requests from metadata only (seed 99): client drawn with prob. proportional to remaining rows; odd steps delete
  the whole client, even steps a random 30% of its remaining rows; identical for every S, seed and policy;
- policies (affected shard j is retrained from a fresh init with its own seed, on its retained data only):
  none            no unlearning (reference, not a valid removal)
  shard_full      15 rounds                              (Affected-Shard Full Retrain)
  shard_fixed     7 rounds                               (Rapid-Retraining-like reduced fixed budget)
  dynamicfu       DynamicFU eqs. 9-21: contribution (theta 1/3 each, lambda 0.5/0.5), eta in [0.2, 1], rho 1, mu 1,
                  omega 0.5/0.5, tau_ctrl 0.01, tau_drop 0.02, R_max 15; controller signals on validation
  utility_stop    round by round; stop at the first round with validation macro-F1 >= F1_0 - 0.005 and FPR <= FPR_0 + 0.005
                  (F1_0, FPR_0: the original ensemble before any request), else 15
  drop_backfill   drop shard j from the ensemble at once; spend 5 rounds per request step rebuilding pending shards FIFO;
                  a shard re-enters only after 15 rounds; the backlog is finished after the last request
- reference: S = 1 single model retrained from scratch (seed s, 15 rounds) on all remaining data after every request;
- exactness: after every step, no included shard may have trained on a deleted row; at the end every included shard
  must equal, bit for bit, a fresh model trained with its seed for its recorded rounds on the final data.
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
from scripts.run_pilot import load_data
from src.data.federated import partition
from src.eval.membership_validation import logit_margin
from src.eval.utility_breakdown import detector_metrics
from src.fl.fedavg_core import model, fedavg, ClientWorker

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
FL = {'rounds': 15, 'local_epochs': 1, 'batch_size': 512, 'lr': 1e-3}; RB = 15; HID = [64, 32]
CLIENTS, ALPHA, SEEDS, SHARDS = 20, .5, [10042 + i for i in range(5)], [4, 10]
K, FRAC, FIXED_R, EPS, BACKFILL = 10, .3, 7, .005, 5
DFU = dict(theta=(1/3, 1/3, 1/3), lam=(.5, .5), eta=(.2, 1.), rho=1., mu=1., omega=(.5, .5), tau_ctrl=.01, tau_drop=.02, Rmax=15)
POLICIES = ['none', 'shard_full', 'shard_fixed', 'dynamicfu', 'utility_stop', 'drop_backfill']
d = load_data(ROOT/'data/processed/connected_v3_seed42'); tr, va, ca = d['train'], d['validation'], d['calibration']
N = len(tr['y']); parts = partition(tr['y'], 42, ALPHA, CLIENTS); owner = np.empty(N, int)
for c, p in enumerate(parts): owner[p] = c
MB = sum(p.numel()*4 for p in model(38, HID, 0, 'cpu').parameters())
p_train = np.bincount(tr['y'], minlength=2)/N

# ---- request stream (metadata only) ----
rng = np.random.default_rng(99); alive0 = np.ones(N, bool); present = set(range(CLIENTS)); stream = []
for k in range(K):
    rem = [(c, parts[c][alive0[parts[c]]]) for c in sorted(present)]; w = np.array([len(r) for _, r in rem], float)
    c, rows = rem[rng.choice(len(rem), p=w/w.sum())]; kind = 'client' if k % 2 == 0 else 'sample'
    f = rows if kind == 'client' else np.sort(rng.choice(rows, max(1, int(len(rows)*FRAC)), replace=False))
    if kind == 'client' or len(f) == len(rows): present.discard(c); kind = 'client'
    alive0[f] = False; stream.append({'step': k + 1, 'client': int(c), 'kind': kind, 'rows': int(len(f)), 'f': f})
(OUT/'stream.json').write_text(json.dumps([{k: v for k, v in r.items() if k != 'f'} for r in stream], indent=1))
print(json.dumps([{k: v for k, v in r.items() if k != 'f'} for r in stream]), flush=True)

def assign(S): perm = np.random.default_rng(7).permutation(CLIENTS); return [sorted(perm[j::S].tolist()) for j in range(S)]

class Trainer:
    """Round-by-round FedAvg of one shard from a fresh init; its first R rounds equal fedavg(init, ws, FL, R, seed)."""
    def __init__(self, idx, seed):
        self.idx, self.seed = idx, seed; self.m = model(38, HID, seed, 'cpu'); self.r = 0
        byc = {}
        for i in idx: byc.setdefault(owner[i], []).append(i)
        self.ws = [ClientWorker(c, tr['X'][np.array(v)], tr['y'][np.array(v)], tr['row_id'][np.array(v)], 'cpu') for c, v in sorted(byc.items())]
        self.rows = len(idx); self.nclients = len(self.ws)
    def step(self):
        self.m, _ = fedavg(self.m, self.ws, FL, 1, self.seed + self.r*10000); self.r += 1

def shard_idx(S_as, j, alive): return np.sort(np.concatenate([parts[c][alive[parts[c]]] for c in S_as[j]]))
def margins(m, sets=('va', 'ca', 'tr')):
    X = {'va': va['X'], 'ca': ca['X'], 'tr': tr['X']}; return {k: logit_margin(m, X[k]) for k in sets}
def ens(sh, key):
    ms = [s['mg'][key] for s in sh if s is not None and s['in']]
    return sum(ms) if ms else np.zeros(len({'va': va, 'ca': ca, 'tr': tr}[key]['y']))
def val_stats(marg):
    met = detector_metrics(va['y'], expit(marg)); loss = float(np.logaddexp(0, -(2*va['y'] - 1)*marg).mean())
    return met['macro_f1'], met['fpr'], loss
def report(sh, alive, present):
    out = {f'cal_{k}': v for k, v in detector_metrics(ca['y'], expit(ens(sh, 'ca'))).items() if k in ('macro_f1', 'fpr', 'recall_attack')}
    mt = ens(sh, 'tr'); f1 = [detector_metrics(tr['y'][p[alive[p]]], expit(mt[p[alive[p]]]))['macro_f1'] for c, p in enumerate(parts) if c in present]
    out.update(client_f1_min=float(np.nanmin(f1)), client_f1_mean=float(np.nanmean(f1))); return out

def make(tr_obj): return {'m': tr_obj.m, 'idx': tr_obj.idx, 'seed': tr_obj.seed, 'rounds': tr_obj.r, 'in': True, 'mg': margins(tr_obj.m)}

def run_rounds(trn, sh, j, rule):
    """Train shard j round by round; rule(r, sh_with_candidate) -> True to stop. Returns the new shard entry."""
    while True:
        trn.step(); cand = {'m': trn.m, 'idx': trn.idx, 'seed': trn.seed, 'rounds': trn.r, 'in': True, 'mg': margins(trn.m, ('va',))}
        tmp = list(sh); tmp[j] = cand
        if rule(trn.r, tmp): break
    return make(trn)

def contributions(sh, S_as, alive_before, present, F1_now):
    """DynamicFU eqs. 9-12 for every present client, on the ensemble before the request."""
    raw = {}
    for c in sorted(present):
        j = next(i for i, a in enumerate(S_as) if c in a); idx = parts[c][alive_before[parts[c]]]
        if sh[j] is None or not len(idx): continue
        w = ClientWorker(c, tr['X'][idx], tr['y'][idx], tr['row_id'][idx], 'cpu')
        st, _, _ = w.train(sh[j]['m'], FL, sh[j]['seed'] + 777); mp = model(38, HID, 0, 'cpu'); mp.load_state_dict(st)
        u = float(torch.cat([(a - b).flatten() for a, b in zip(mp.state_dict().values(), sh[j]['m'].state_dict().values())]).norm())
        pi = np.bincount(tr['y'][idx], minlength=2)/len(idx); dd = DFU['lam'][0]*jensenshannon(pi, p_train)**2 + DFU['lam'][1]*pi[1]
        tmp = list(sh); tmp[j] = {**sh[j], 'mg': {'va': logit_margin(mp, va['X'])}}
        q = abs(val_stats(ens(tmp, 'va'))[0] - F1_now); raw[c] = (u, dd, q)
    a = np.array(list(raw.values())); norm = (a - a.min(0))/(a.max(0) - a.min(0) + 1e-12)
    return {c: float(np.dot(DFU['theta'], n)) for c, n in zip(raw, norm)}

rows, checks = [], []
for S in SHARDS:
    S_as = assign(S); shard_of = {c: j for j, a in enumerate(S_as) for c in a}
    for s in SEEDS:
        base = []
        for j in range(S):
            t = Trainer(shard_idx(S_as, j, np.ones(N, bool)), s + 1000*j)
            for _ in range(RB): t.step()
            base.append(make(t))
        F1_0, FPR_0, _ = val_stats(ens(base, 'va'))
        for pol in POLICIES:
            sh = [dict(b) for b in base]; alive = np.ones(N, bool); present = set(range(CLIENTS)); pending = []
            cost = dict(rows_x_rounds=0, rounds=0, bytes=0, seconds=0.)
            def charge(trn, r0, t0):
                n = trn.r - r0; cost['rows_x_rounds'] += trn.rows*n; cost['rounds'] += n; cost['bytes'] += 2*MB*trn.nclients*n
                cost['seconds'] += time.perf_counter() - t0
            for req in stream:
                alive_before = alive.copy(); alive[req['f']] = False
                if req['kind'] == 'client': present.discard(req['client'])
                j = shard_of[req['client']]; idx = shard_idx(S_as, j, alive); t0 = time.perf_counter(); extra = {}
                if pol != 'none' and not len(idx):                                  # shard has no data left
                    sh[j] = None; pending = [p for p in pending if p[0] != j]
                elif pol in ('shard_full', 'shard_fixed'):
                    trn = Trainer(idx, s + 1000*j); R = RB if pol == 'shard_full' else FIXED_R
                    for _ in range(R): trn.step()
                    sh[j] = make(trn); charge(trn, 0, t0)
                elif pol == 'utility_stop':
                    trn = Trainer(idx, s + 1000*j)
                    def rule(r, tmp):
                        f1, fpr, _ = val_stats(ens(tmp, 'va')); return r >= RB or (f1 >= F1_0 - EPS and fpr <= FPR_0 + EPS)
                    sh[j] = run_rounds(trn, sh, j, rule); charge(trn, 0, t0)
                elif pol == 'dynamicfu':
                    F1_now = val_stats(ens(sh, 'va'))[0]; tc = time.perf_counter()
                    I = contributions(sh, S_as, alive_before, present | {req['client']}, F1_now); extra['contrib_seconds'] = time.perf_counter() - tc
                    eta = float(np.clip(DFU['eta'][0] + (DFU['eta'][1] - DFU['eta'][0])*I.get(req['client'], 0.), *DFU['eta']))
                    trn = Trainer(idx, s + 1000*j); init = {'m': trn.m, 'in': True, 'mg': {'va': logit_margin(trn.m, va['X'])}}
                    tmp = list(sh); tmp[j] = init; prev_f1, _, prev_loss = val_stats(ens(tmp, 'va')); st = {'eta': eta}
                    extra.update(contribution=I.get(req['client'], 0.), R0=max(1, round(RB*eta)))
                    def rule(r, tmp2):
                        nonlocal_prev = st
                        f1, _, loss = val_stats(ens(tmp2, 'va'))
                        G = max(0., nonlocal_prev.get('loss', prev_loss) - loss)/(nonlocal_prev.get('loss', prev_loss) + 1e-12)
                        stab = abs(f1 - nonlocal_prev.get('f1', prev_f1)); ctrl = DFU['omega'][0]*G + DFU['omega'][1]*stab
                        drop = max(0., F1_now - f1); nonlocal_prev['eta'] = float(np.clip(nonlocal_prev['eta'] + DFU['rho']*ctrl - DFU['mu']*drop, *DFU['eta']))
                        Rr = max(1, round(RB*nonlocal_prev['eta'])); nonlocal_prev.update(loss=loss, f1=f1)
                        stop = (r >= Rr and ctrl <= DFU['tau_ctrl']) or drop >= DFU['tau_drop'] or r >= DFU['Rmax']
                        if stop: nonlocal_prev['reason'] = 'drop' if drop >= DFU['tau_drop'] else ('rmax' if r >= DFU['Rmax'] else 'ctrl')
                        return stop
                    sh[j] = run_rounds(trn, sh, j, rule); charge(trn, 0, t0); extra['stop_reason'] = st.get('reason')
                elif pol == 'drop_backfill':
                    if sh[j] is not None: sh[j] = {**sh[j], 'in': False}
                    pending = [p for p in pending if p[0] != j] + [(j, Trainer(idx, s + 1000*j))]
                    budget = BACKFILL
                    while budget and pending:
                        jj, trn = pending[0]; r0 = trn.r; t1 = time.perf_counter(); n = min(budget, RB - trn.r)
                        for _ in range(n): trn.step()
                        charge(trn, r0, t1); budget -= n
                        if trn.r >= RB: sh[jj] = make(trn); pending.pop(0)
                    extra['pending'] = len(pending)
                exact = all(not (~alive[e['idx']]).any() for e in sh if e is not None and e['in'])
                rows.append({'S': S, 'seed': s, 'policy': pol, 'step': req['step'], 'kind': req['kind'], 'exact': exact,
                             **cost, **report(sh, alive, present), **extra})
            if pol == 'drop_backfill':
                for jj, trn in pending:
                    r0 = trn.r; t1 = time.perf_counter()
                    while trn.r < RB: trn.step()
                    charge(trn, r0, t1); sh[jj] = make(trn)
                rows.append({'S': S, 'seed': s, 'policy': pol, 'step': K + 1, 'kind': 'backlog_done', 'exact': True, **cost, **report(sh, alive, present)})
            if pol != 'none':   # bit-for-bit check against fresh training on the final data
                ok = []
                for j, e in enumerate(sh):
                    if e is None or not e['in']: continue
                    fresh = Trainer(shard_idx(S_as, j, alive), e['seed'])
                    for _ in range(e['rounds']): fresh.step()
                    ok.append(all(torch.equal(a, b) for a, b in zip(fresh.m.state_dict().values(), e['m'].state_dict().values())))
                checks.append({'S': S, 'seed': s, 'policy': pol, 'shards_checked': len(ok), 'bit_exact': all(ok)})
        print(f'S={S} seed={s} done', flush=True)
# reference: single model (S = 1) retrained from scratch after every request
for s in SEEDS:
    alive = np.ones(N, bool); present = set(range(CLIENTS)); cum = dict(rows_x_rounds=0, rounds=0, bytes=0, seconds=0.)
    for req in stream:
        alive[req['f']] = False
        if req['kind'] == 'client': present.discard(req['client'])
        t0 = time.perf_counter(); trn = Trainer(np.flatnonzero(alive), s)
        for _ in range(RB): trn.step()
        cum['rows_x_rounds'] += trn.rows*RB; cum['rounds'] += RB; cum['bytes'] += 2*MB*trn.nclients*RB; cum['seconds'] += time.perf_counter() - t0
        rows.append({'S': 1, 'seed': s, 'policy': 'full_retrain_single', 'step': req['step'], 'kind': req['kind'], 'exact': True, **cum,
                     **report([make(trn)], alive, present)})
    print(f'reference seed={s} done', flush=True)
pd.DataFrame(rows).to_csv(OUT/'steps.csv', index=False); pd.DataFrame(checks).to_csv(OUT/'exactness.csv', index=False)
print('DONE', flush=True)
