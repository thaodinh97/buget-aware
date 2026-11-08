"""Run 2 of the v2 shard protocol (docs/shard_v2_plan.md): sequential deletion stream with sharded exact unlearning.
Usage: python run_shard_stream_v2.py OUT_DIR

Fixed before running (design from scripts/shard/shard_v2.py, SHARD_DESIGN, default 'v2'):
- data SHARD_DATA_DIR (default connected_v3_auxfit_seed42), content hash checked; policies may read VALIDATION; reported
  metrics on CALIBRATION; test unused;
- 20 clients, Dirichlet alpha 0.5, partition seed 42; S in {4, 5} by seeded permutation (seed 7), every shard >= 4 clients;
- FedAvg per shard with the 'standard' config: R_base 20 rounds, 2 local epochs, Adam 1e-2, batch 512; FIXED_R 10; BACKFILL 7;
  DynamicFU R_max 20; ensemble = sum of shard margins; shard j seed s + 1000 j; training seeds 50042..50051;
- stream: 10 requests from metadata only (seed 99), identical to v1;
- shard structure (shard_v2.structure_plan, the same for every policy and seed): a whole-client deletion that leaves a shard
  with < 3 present clients merges its remaining clients into the active shard with the fewest present clients (ties: lowest
  index); the merged shard keeps that index and seed and is retrained from a fresh init; a shard left with no client is removed;
- policies (the shard to retrain = the affected or the merged shard, from a fresh init with its own seed, on current data only):
  none, shard_full, shard_fixed, dynamicfu, drop_backfill   as in v1, on the v2 structure
  utility_stop_v1  v1 rule: round by round, stop at the first round with ok(), else at R_base; nothing else (descriptive only)
  utility_stop     v2 rule: (1) as utility_stop_v1; (2) after every request - also a removal or a merge - if not ok(): fallback:
                   give one more round to the included shard with the fewest rounds below R_base (ties: lowest index),
                   continuing its own seeded run, and re-check; repeat until ok() or every included shard has R_base rounds
                   (that end state equals shard_full's); condition_met records whether ok() holds at the end of the step
  ok(): validation macro-F1 >= F1_0 - 0.005 and validation FPR <= FPR_0 + 0.005 (F1_0, FPR_0: original ensemble)
- reference: S = 1 single model retrained from scratch (seed s, R_base rounds) on all remaining data after every request;
- exactness: after every step no included shard has trained on a deleted row (exact) and every included shard is trained on
  exactly the current data of its shard (structure_ok); at the end every included shard equals, bit for bit, a fresh model
  trained with its seed for its recorded rounds on the final data of its shard, and the included shards are the active ones.
"""
import copy, json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
from src.data.federated import partition
from src.eval.membership_validation import logit_margin
from src.eval.utility_breakdown import detector_metrics
from src.fl.fedavg_core import model, fedavg, ClientWorker
from scripts.shard.shard_v2 import start_run, assign, check_design, structure_plan

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
DESIGN_NAME, DESIGN, CFG, d = start_run(OUT, __file__)
FL = CFG['fl']; RB = FL['rounds']; HID = [64, 32]
CLIENTS, ALPHA, SEEDS, SHARDS = 20, .5, DESIGN['seeds'], DESIGN['shards_run2']
K, FRAC, FIXED_R, EPS, BACKFILL = 10, .3, CFG['shard_fixed_rounds'], .005, CFG['backfill_rounds']
DFU = dict(theta=(1/3, 1/3, 1/3), lam=(.5, .5), eta=(.2, 1.), rho=1., mu=1., omega=(.5, .5), tau_ctrl=.01, tau_drop=.02, Rmax=RB)
POLICIES = ['none', 'shard_full', 'shard_fixed', 'dynamicfu', 'utility_stop', 'utility_stop_v1', 'drop_backfill']
tr, va, ca = d['train'], d['validation'], d['calibration']; DIM = tr['X'].shape[1]
N = len(tr['y']); parts = partition(tr['y'], 42, ALPHA, CLIENTS); owner = np.empty(N, int)
for c, p in enumerate(parts): owner[p] = c
MB = sum(p.numel()*4 for p in model(DIM, HID, 0, 'cpu').parameters())
p_train = np.bincount(tr['y'], minlength=2)/N

# ---- request stream (metadata only; identical to v1) ----
rng = np.random.default_rng(99); alive0 = np.ones(N, bool); present = set(range(CLIENTS)); stream = []
for k in range(K):
    rem = [(c, parts[c][alive0[parts[c]]]) for c in sorted(present)]; w = np.array([len(r) for _, r in rem], float)
    c, rows = rem[rng.choice(len(rem), p=w/w.sum())]; kind = 'client' if k % 2 == 0 else 'sample'
    f = rows if kind == 'client' else np.sort(rng.choice(rows, max(1, int(len(rows)*FRAC)), replace=False))
    if kind == 'client' or len(f) == len(rows): present.discard(c); kind = 'client'
    alive0[f] = False; stream.append({'step': k + 1, 'client': int(c), 'kind': kind, 'rows': int(len(f)), 'f': f})
(OUT/'stream.json').write_text(json.dumps([{k: v for k, v in r.items() if k != 'f'} for r in stream], indent=1))
print(json.dumps([{k: v for k, v in r.items() if k != 'f'} for r in stream]), flush=True)

class Trainer:
    """Round-by-round FedAvg of one shard from a fresh init; its first R rounds equal fedavg(init, ws, FL, R, seed)."""
    def __init__(self, idx, seed):
        self.idx, self.seed = idx, seed; self.m = model(DIM, HID, seed, 'cpu'); self.r = 0
        byc = {}
        for i in idx: byc.setdefault(owner[i], []).append(i)
        self.ws = [ClientWorker(c, tr['X'][np.array(v)], tr['y'][np.array(v)], tr['row_id'][np.array(v)], 'cpu') for c, v in sorted(byc.items())]
        self.rows = len(idx); self.nclients = len(self.ws)
    def step(self):
        self.m, _ = fedavg(self.m, self.ws, FL, 1, self.seed + self.r*10000); self.r += 1

def resume(e):
    """Continue the seeded run of shard entry e; equal to a fresh Trainer(e['idx'], e['seed']) after e['rounds'] rounds."""
    t = Trainer(e['idx'], e['seed']); t.m = copy.deepcopy(e['m']); t.r = e['rounds']; return t

def shard_idx(A, j, alive):
    blocks = [parts[c][alive[parts[c]]] for c in A[j]]
    return np.sort(np.concatenate(blocks)) if blocks else np.empty(0, np.int64)
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
def structure_ok(sh, A, alive):
    return all(A[j] and np.array_equal(e['idx'], shard_idx(A, j, alive)) for j, e in enumerate(sh) if e is not None and e['in'])

def make(tr_obj): return {'m': tr_obj.m, 'idx': tr_obj.idx, 'seed': tr_obj.seed, 'rounds': tr_obj.r, 'in': True, 'mg': margins(tr_obj.m)}

def run_rounds(trn, sh, j, rule):
    """Train shard j round by round; rule(r, sh_with_candidate) -> True to stop. Returns the new shard entry."""
    while True:
        trn.step(); cand = {'m': trn.m, 'idx': trn.idx, 'seed': trn.seed, 'rounds': trn.r, 'in': True, 'mg': margins(trn.m, ('va',))}
        tmp = list(sh); tmp[j] = cand
        if rule(trn.r, tmp): break
    return make(trn)

def contributions(sh, A, alive_before, present, F1_now):
    """DynamicFU eqs. 9-12 for every present client, on the ensemble before the request (structure A before the request)."""
    raw = {}
    for c in sorted(present):
        j = next(i for i, a in enumerate(A) if c in a); idx = parts[c][alive_before[parts[c]]]
        if sh[j] is None or not len(idx): continue
        w = ClientWorker(c, tr['X'][idx], tr['y'][idx], tr['row_id'][idx], 'cpu')
        st, _, _ = w.train(sh[j]['m'], FL, sh[j]['seed'] + 777); mp = model(DIM, HID, 0, 'cpu'); mp.load_state_dict(st)
        u = float(torch.cat([(a - b).flatten() for a, b in zip(mp.state_dict().values(), sh[j]['m'].state_dict().values())]).norm())
        pi = np.bincount(tr['y'][idx], minlength=2)/len(idx); dd = DFU['lam'][0]*jensenshannon(pi, p_train)**2 + DFU['lam'][1]*pi[1]
        tmp = list(sh); tmp[j] = {**sh[j], 'mg': {'va': logit_margin(mp, va['X'])}}
        q = abs(val_stats(ens(tmp, 'va'))[0] - F1_now); raw[c] = (u, dd, q)
    a = np.array(list(raw.values())); norm = (a - a.min(0))/(a.max(0) - a.min(0) + 1e-12)
    return {c: float(np.dot(DFU['theta'], n)) for c, n in zip(raw, norm)}

rows, checks, plans = [], [], {}
for S in SHARDS:
    A0 = assign(S, CLIENTS); check_design(A0, DESIGN['min_clients_design'])
    plan = structure_plan(A0, stream, DESIGN['merge_below']); plans[S] = plan
    print(f'S={S} structure events:', [(p['step'], p['event']) for p in plan if p['event'] != 'retrain'], flush=True)
    for s in SEEDS:
        base = []
        for j in range(S):
            t = Trainer(shard_idx(A0, j, np.ones(N, bool)), s + 1000*j)
            for _ in range(RB): t.step()
            base.append(make(t))
        F1_0, FPR_0, _ = val_stats(ens(base, 'va'))
        def ok(tmp):
            f1, fpr, _ = val_stats(ens(tmp, 'va')); return f1 >= F1_0 - EPS and fpr <= FPR_0 + EPS
        for pol in POLICIES:
            sh = [dict(b) for b in base]; alive = np.ones(N, bool); present = set(range(CLIENTS)); pending = []
            cost = dict(rows_x_rounds=0, rounds=0, bytes=0, seconds=0.)
            def charge(trn, r0, t0):
                n = trn.r - r0; cost['rows_x_rounds'] += trn.rows*n; cost['rounds'] += n; cost['bytes'] += 2*MB*trn.nclients*n
                cost['seconds'] += time.perf_counter() - t0
            for req, pl in zip(stream, plan):
                alive_before = alive.copy(); alive[req['f']] = False
                if req['kind'] == 'client': present.discard(req['client'])
                A, j = pl['assign_after'], pl['retrain']; t0 = time.perf_counter(); extra = {'event': pl['event']}
                if pol == 'dynamicfu' and j is not None:       # contributions on the ensemble before the request
                    F1_now = val_stats(ens(sh, 'va'))[0]; tc = time.perf_counter()
                    I = contributions(sh, pl['assign_before'], alive_before, present | {req['client']}, F1_now)
                    extra['contrib_seconds'] = time.perf_counter() - tc
                if pol != 'none':                                # removed / merged-away shards leave the ensemble
                    for jr in pl['retired']: sh[jr] = None; pending = [p for p in pending if p[0] != jr]
                idx = shard_idx(A, j, alive) if j is not None else None
                if pol in ('shard_full', 'shard_fixed') and j is not None:
                    trn = Trainer(idx, s + 1000*j); R = RB if pol == 'shard_full' else FIXED_R
                    for _ in range(R): trn.step()
                    sh[j] = make(trn); charge(trn, 0, t0)
                elif pol in ('utility_stop', 'utility_stop_v1'):
                    if j is not None:
                        trn = Trainer(idx, s + 1000*j)
                        sh[j] = run_rounds(trn, sh, j, lambda r, tmp: r >= RB or ok(tmp)); charge(trn, 0, t0)
                        extra['phase1_rounds'] = trn.r
                    if pol == 'utility_stop':
                        fb = 0
                        while not ok(sh):
                            cand = [(e['rounds'], jj) for jj, e in enumerate(sh) if e is not None and e['in'] and e['rounds'] < RB]
                            if not cand: break
                            jj = min(cand)[1]; trn = resume(sh[jj]); t1 = time.perf_counter(); trn.step()
                            charge(trn, trn.r - 1, t1); sh[jj] = make(trn); fb += 1
                        extra.update(fallback_rounds=fb, condition_met=bool(ok(sh)))
                elif pol == 'dynamicfu' and j is not None:
                    eta = float(np.clip(DFU['eta'][0] + (DFU['eta'][1] - DFU['eta'][0])*I.get(req['client'], 0.), *DFU['eta']))
                    trn = Trainer(idx, s + 1000*j); init = {'m': trn.m, 'in': True, 'mg': {'va': logit_margin(trn.m, va['X'])}}
                    tmp = list(sh); tmp[j] = init; prev_f1, _, prev_loss = val_stats(ens(tmp, 'va')); st = {'eta': eta}
                    extra.update(contribution=I.get(req['client'], 0.), R0=max(1, round(RB*eta)))
                    def rule(r, tmp2):
                        f1, _, loss = val_stats(ens(tmp2, 'va'))
                        G = max(0., st.get('loss', prev_loss) - loss)/(st.get('loss', prev_loss) + 1e-12)
                        stab = abs(f1 - st.get('f1', prev_f1)); ctrl = DFU['omega'][0]*G + DFU['omega'][1]*stab
                        drop = max(0., F1_now - f1); st['eta'] = float(np.clip(st['eta'] + DFU['rho']*ctrl - DFU['mu']*drop, *DFU['eta']))
                        Rr = max(1, round(RB*st['eta'])); st.update(loss=loss, f1=f1)
                        stop = (r >= Rr and ctrl <= DFU['tau_ctrl']) or drop >= DFU['tau_drop'] or r >= DFU['Rmax']
                        if stop: st['reason'] = 'drop' if drop >= DFU['tau_drop'] else ('rmax' if r >= DFU['Rmax'] else 'ctrl')
                        return stop
                    sh[j] = run_rounds(trn, sh, j, rule); charge(trn, 0, t0); extra['stop_reason'] = st.get('reason')
                elif pol == 'drop_backfill':
                    if j is not None:
                        if sh[j] is not None: sh[j] = {**sh[j], 'in': False}
                        pending = [p for p in pending if p[0] != j] + [(j, Trainer(idx, s + 1000*j))]
                    budget = BACKFILL
                    while budget and pending:
                        jj, trn = pending[0]; r0 = trn.r; t1 = time.perf_counter(); n = min(budget, RB - trn.r)
                        for _ in range(n): trn.step()
                        charge(trn, r0, t1); budget -= n
                        if trn.r >= RB: sh[jj] = make(trn); pending.pop(0)
                    extra['pending'] = len(pending)
                inc = [e for e in sh if e is not None and e['in']]
                rows.append({'S': S, 'seed': s, 'policy': pol, 'step': req['step'], 'kind': req['kind'],
                             'exact': all(not (~alive[e['idx']]).any() for e in inc),
                             'structure_ok': structure_ok(sh, A, alive) if pol != 'none' else np.nan, 'shards_included': len(inc),
                             **cost, **report(sh, alive, present), **extra})
            A_fin = plan[-1]['assign_after']
            if pol == 'drop_backfill':
                for jj, trn in pending:
                    r0 = trn.r; t1 = time.perf_counter()
                    while trn.r < RB: trn.step()
                    charge(trn, r0, t1); sh[jj] = make(trn)
                inc = [e for e in sh if e is not None and e['in']]
                rows.append({'S': S, 'seed': s, 'policy': pol, 'step': K + 1, 'kind': 'backlog_done',
                             'exact': all(not (~alive[e['idx']]).any() for e in inc), 'structure_ok': structure_ok(sh, A_fin, alive),
                             'shards_included': len(inc), **cost, **report(sh, alive, present)})
            if pol != 'none':   # bit-for-bit check against fresh training on the final data of each shard
                bits = []
                for jj, e in enumerate(sh):
                    if e is None or not e['in']: continue
                    fresh = Trainer(shard_idx(A_fin, jj, alive), e['seed'])
                    for _ in range(e['rounds']): fresh.step()
                    bits.append(all(torch.equal(a, b) for a, b in zip(fresh.m.state_dict().values(), e['m'].state_dict().values())))
                active = sorted(jj for jj, a in enumerate(A_fin) if a)
                included = sorted(jj for jj, e in enumerate(sh) if e is not None and e['in'])
                checks.append({'S': S, 'seed': s, 'policy': pol, 'shards_checked': len(bits), 'bit_exact': all(bits),
                               'included_equals_active': included == active, 'structure_ok_end': structure_ok(sh, A_fin, alive)})
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
        rows.append({'S': 1, 'seed': s, 'policy': 'full_retrain_single', 'step': req['step'], 'kind': req['kind'], 'exact': True,
                     'structure_ok': True, 'shards_included': 1, **cum, **report([make(trn)], alive, present)})
    print(f'reference seed={s} done', flush=True)
pd.DataFrame(rows).to_csv(OUT/'steps.csv', index=False); pd.DataFrame(checks).to_csv(OUT/'exactness.csv', index=False)
(OUT/'structure.json').write_text(json.dumps({str(S): p for S, p in plans.items()}, indent=1))
print('DONE', flush=True)
