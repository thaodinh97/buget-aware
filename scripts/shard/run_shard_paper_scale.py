"""DynamicFU-scale check (exploratory, not preregistered; results in results/shard_exploration): 50 clients, 10 shards, 20 rounds, 2 local epochs, Adam 1e-2.

Usage: python paper_scale.py OUT_DIR ALPHA
Fixed before running:
- data connected_v3_seed42; policies may read VALIDATION; reported metrics on CALIBRATION; test unused;
- partition: the repo's label-Dirichlet procedure (same RNG calls as src/data/federated.partition) with 50 clients, but a client
  only has to hold >= 10 rows (single-class clients allowed; needed for alpha 0.1); seed = first seed >= 42 that is valid;
- 10 shards of 5 clients by seeded permutation (seed 7); shard j seed s + 1000 j; ensemble = sum of shard margins;
- model: the repo MLP 38-64-32-2; training seeds 10042..10046;
- Part A, sequential stream: 10 requests from metadata (seed 99), odd steps delete a whole client, even steps 30% of one;
  policies none / shard_full (20 rounds) / shard_fixed (10 rounds, Rapid-Retraining-like) / dynamicfu / utility_stop / drop_backfill
  (7 rounds of rebuild per step); reference: single model retrained from scratch (20 rounds) after every request;
- Part B, the paper's protocol: contribution of all 50 clients on the trained ensemble (DynamicFU eqs. 9-12); 3 clients drawn
  (seed 123) from each tercile (high / medium / low) -> 9 single client-level deletions from the original ensemble;
  methods none / full_retrain (all 10 shards, 20 rounds) / affected_shard (20) / rapid (10) / dynamicfu / utility_stop;
- DynamicFU: theta 1/3 each, lambda 0.5/0.5, eta in [0.2, 1], R_base = R_max = 20, rho 1, mu 1, omega 0.5/0.5, tau_ctrl 0.01,
  tau_drop 0.02; controller signals on validation (no per-shard validation split exists);
- utility_stop: first round with validation macro-F1 >= F1_ref - 0.005 and FPR <= FPR_ref + 0.005 (ref = ensemble before
  any request), else 20;
- exactness: per step, no included shard trained on a deleted row; at the end, bit-for-bit equality with fresh training.
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from scipy.special import expit
from scipy.spatial.distance import jensenshannon
from scripts.run_pilot import load_data
from src.eval.membership_validation import logit_margin
from src.eval.utility_breakdown import detector_metrics
from src.fl.fedavg_core import model, fedavg, ClientWorker

OUT = Path(sys.argv[1]); ALPHA = float(sys.argv[2]); OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(1); torch.use_deterministic_algorithms(True)
FL = {'rounds': 20, 'local_epochs': 2, 'batch_size': 512, 'lr': 1e-2}; RB = 20; HID = [64, 32]
CLIENTS, S, SEEDS = 50, 10, [10042 + i for i in range(5)]
K, FRAC, FIXED_R, EPS, BACKFILL = 10, .3, 10, .005, 7
DFU = dict(theta=(1/3, 1/3, 1/3), lam=(.5, .5), eta=(.2, 1.), rho=1., mu=1., omega=(.5, .5), tau_ctrl=.01, tau_drop=.02, Rmax=20)
d = load_data(ROOT/'data/processed/connected_v3_seed42'); tr, va, ca = d['train'], d['validation'], d['calibration']
N = len(tr['y']); p_train = np.bincount(tr['y'], minlength=2)/N
MB = sum(p.numel()*4 for p in model(38, HID, 0, 'cpu').parameters())

def dirichlet_partition(y, seed, alpha, clients, min_rows=10):
    rng = np.random.default_rng(seed); result = [[] for _ in range(clients)]
    for label in [0, 1]:
        idx = rng.permutation(np.flatnonzero(y == label)); lengths = rng.multinomial(len(idx), rng.dirichlet(np.full(clients, alpha)))
        for dest, part in zip(result, np.split(idx, np.cumsum(lengths)[:-1])): dest.extend(part.tolist())
    result = [np.array(sorted(x), dtype=np.int64) for x in result]
    if min(len(r) for r in result) < min_rows: raise ValueError('too small client')
    return result
def equal_dirichlet_partition(y, seed, alpha, clients):
    """Alternative scheme (used for alpha 0.1, where the per-class scheme leaves empty clients for every seed tried):
    equal client sizes, each client's class mix ~ Dirichlet(alpha); when a class pool runs out the rest comes from the other;
    the few leftover rows are dealt round-robin."""
    rng = np.random.default_rng(seed); pools = {c: list(rng.permutation(np.flatnonzero(y == c))) for c in [0, 1]}
    size = len(y)//clients; result = []
    for _ in range(clients):
        p0 = rng.dirichlet(np.full(2, alpha))[0]; k0 = min(int(round(p0*size)), len(pools[0])); k1 = min(size - k0, len(pools[1]))
        k0 = min(size - k1, len(pools[0])); result.append(pools[0][:k0] + pools[1][:k1]); pools[0] = pools[0][k0:]; pools[1] = pools[1][k1:]
    for i, r in enumerate(pools[0] + pools[1]): result[i % clients].append(r)
    return [np.array(sorted(x), dtype=np.int64) for x in result]
SCHEME = sys.argv[3] if len(sys.argv) > 3 else 'per_class'
pseed = 42
while SCHEME == 'per_class':
    try: parts = dirichlet_partition(tr['y'], pseed, ALPHA, CLIENTS); break
    except ValueError: pseed += 1
if SCHEME == 'equal': parts = equal_dirichlet_partition(tr['y'], pseed, ALPHA, CLIENTS)
assert len(np.unique(np.concatenate(parts))) == N
owner = np.empty(N, int)
for c, p in enumerate(parts): owner[p] = c
perm = np.random.default_rng(7).permutation(CLIENTS); S_as = [sorted(perm[j::S].tolist()) for j in range(S)]
shard_of = {c: j for j, a in enumerate(S_as) for c in a}
meta = {'alpha': ALPHA, 'scheme': SCHEME, 'partition_seed': pseed, 'client_rows': [len(p) for p in parts],
        'client_normal': [int((tr['y'][p] == 0).sum()) for p in parts], 'single_class_clients': int(sum(len(set(tr['y'][p])) == 1 for p in parts)),
        'shards': S_as, 'shard_rows': [int(sum(len(parts[c]) for c in a)) for a in S_as]}

class Trainer:
    def __init__(self, idx, seed):
        self.idx, self.seed = idx, seed; self.m = model(38, HID, seed, 'cpu'); self.r = 0; byc = {}
        for i in idx: byc.setdefault(owner[i], []).append(i)
        self.ws = [ClientWorker(c, tr['X'][np.array(v)], tr['y'][np.array(v)], tr['row_id'][np.array(v)], 'cpu') for c, v in sorted(byc.items())]
        self.rows = len(idx); self.nclients = len(self.ws)
    def step(self): self.m, _ = fedavg(self.m, self.ws, FL, 1, self.seed + self.r*10000); self.r += 1

def shard_idx(j, alive): return np.sort(np.concatenate([parts[c][alive[parts[c]]] for c in S_as[j]]))
def margins(m, sets=('va', 'ca', 'tr')):
    X = {'va': va['X'], 'ca': ca['X'], 'tr': tr['X']}; return {k: logit_margin(m, X[k]) for k in sets}
LEN = {'va': len(va['y']), 'ca': len(ca['y']), 'tr': N}
def ens(sh, key):
    ms = [e['mg'][key] for e in sh if e is not None and e['in']]; return sum(ms) if ms else np.zeros(LEN[key])
def val_stats(mg):
    met = detector_metrics(va['y'], expit(mg)); return met['macro_f1'], met['fpr'], float(np.logaddexp(0, -(2*va['y'] - 1)*mg).mean())
def report(sh, alive, present):
    mc = ens(sh, 'ca'); out = {f'cal_{k}': v for k, v in detector_metrics(ca['y'], expit(mc)).items() if k in ('macro_f1', 'fpr', 'recall_attack')}
    out['cal_acc'] = float(((mc >= 0) == (ca['y'] == 1)).mean())
    mt = ens(sh, 'tr'); f1 = [detector_metrics(tr['y'][p[alive[p]]], expit(mt[p[alive[p]]]))['macro_f1'] for c, p in enumerate(parts) if c in present and alive[p].any()]
    out.update(client_f1_min=float(np.nanmin(f1)), client_f1_mean=float(np.nanmean(f1))); return out
def make(t): return {'m': t.m, 'idx': t.idx, 'seed': t.seed, 'rounds': t.r, 'in': True, 'mg': margins(t.m)}
def run_rounds(t, sh, j, rule):
    while True:
        t.step(); tmp = list(sh); tmp[j] = {'m': t.m, 'in': True, 'mg': margins(t.m, ('va',))}
        if rule(t.r, tmp): return make(t)

def contributions(sh, alive, cands, F1_now):
    raw = {}
    for c in sorted(cands):
        j = shard_of[c]; idx = parts[c][alive[parts[c]]]
        if sh[j] is None or not len(idx): continue
        w = ClientWorker(c, tr['X'][idx], tr['y'][idx], tr['row_id'][idx], 'cpu'); st, _, _ = w.train(sh[j]['m'], FL, sh[j]['seed'] + 777)
        mp = model(38, HID, 0, 'cpu'); mp.load_state_dict(st)
        u = float(torch.cat([(a - b).flatten() for a, b in zip(mp.state_dict().values(), sh[j]['m'].state_dict().values())]).norm())
        pi = np.bincount(tr['y'][idx], minlength=2)/len(idx); dd = DFU['lam'][0]*jensenshannon(pi, p_train)**2 + DFU['lam'][1]*pi[1]
        tmp = list(sh); tmp[j] = {**sh[j], 'mg': {'va': logit_margin(mp, va['X'])}}; q = abs(val_stats(ens(tmp, 'va'))[0] - F1_now)
        raw[c] = (u, dd, q)
    a = np.array(list(raw.values())); nrm = (a - a.min(0))/(a.max(0) - a.min(0) + 1e-12)
    return {c: float(np.dot(DFU['theta'], n)) for c, n in zip(raw, nrm)}, int(sum(len(parts[c][alive[parts[c]]]) for c in raw))

def dynamicfu(sh, j, idx, seed, I_u, F1_now):
    eta = float(np.clip(DFU['eta'][0] + (DFU['eta'][1] - DFU['eta'][0])*I_u, *DFU['eta'])); t = Trainer(idx, seed)
    tmp = list(sh); tmp[j] = {'m': t.m, 'in': True, 'mg': {'va': logit_margin(t.m, va['X'])}}
    f1p, _, lp = val_stats(ens(tmp, 'va')); st = {'eta': eta, 'f1': f1p, 'loss': lp}
    def rule(r, tmp2):
        f1, _, loss = val_stats(ens(tmp2, 'va')); G = max(0., st['loss'] - loss)/(st['loss'] + 1e-12); ctrl = DFU['omega'][0]*G + DFU['omega'][1]*abs(f1 - st['f1'])
        drop = max(0., F1_now - f1); st['eta'] = float(np.clip(st['eta'] + DFU['rho']*ctrl - DFU['mu']*drop, *DFU['eta'])); Rr = max(1, round(RB*st['eta']))
        st.update(f1=f1, loss=loss); stop = (r >= Rr and ctrl <= DFU['tau_ctrl']) or drop >= DFU['tau_drop'] or r >= DFU['Rmax']
        if stop: st['reason'] = 'drop' if drop >= DFU['tau_drop'] else ('rmax' if r >= DFU['Rmax'] else 'ctrl')
        return stop
    e = run_rounds(t, sh, j, rule); return e, t, {'R0': max(1, round(RB*eta)), 'stop_reason': st['reason'], 'contribution': I_u}

def utility_stop(sh, j, idx, seed, F1_ref, FPR_ref):
    t = Trainer(idx, seed)
    def rule(r, tmp):
        f1, fpr, _ = val_stats(ens(tmp, 'va')); return r >= RB or (f1 >= F1_ref - EPS and fpr <= FPR_ref + EPS)
    return run_rounds(t, sh, j, rule), t

def bit_exact(sh, alive):
    ok = []
    for j, e in enumerate(sh):
        if e is None or not e['in']: continue
        f = Trainer(shard_idx(j, alive), e['seed'])
        for _ in range(e['rounds']): f.step()
        ok.append(all(torch.equal(a, b) for a, b in zip(f.m.state_dict().values(), e['m'].state_dict().values())))
    return all(ok), len(ok)

# ---- stream (metadata only) ----
rng = np.random.default_rng(99); a0 = np.ones(N, bool); pres = set(range(CLIENTS)); stream = []
for k in range(K):
    rem = [(c, parts[c][a0[parts[c]]]) for c in sorted(pres)]; w = np.array([len(r) for _, r in rem], float)
    c, rows = rem[rng.choice(len(rem), p=w/w.sum())]; kind = 'client' if k % 2 == 0 else 'sample'
    f = rows if kind == 'client' else np.sort(rng.choice(rows, max(1, int(len(rows)*FRAC)), replace=False))
    if kind == 'client' or len(f) == len(rows): pres.discard(c); kind = 'client'
    a0[f] = False; stream.append({'step': k + 1, 'client': int(c), 'kind': kind, 'rows': int(len(f)), 'f': f})
meta['stream'] = [{k: v for k, v in r.items() if k != 'f'} for r in stream]; meta['stream_deleted_fraction'] = float(1 - a0.mean())
(OUT/'meta.json').write_text(json.dumps(meta, indent=1)); print(json.dumps({k: meta[k] for k in ('alpha', 'partition_seed', 'single_class_clients', 'stream_deleted_fraction')}), flush=True)

rowsA, rowsB, checks = [], [], []
for s in SEEDS:
    t0 = time.perf_counter(); base = []
    for j in range(S):
        t = Trainer(shard_idx(j, np.ones(N, bool)), s + 1000*j)
        for _ in range(RB): t.step()
        base.append(make(t))
    T_full = time.perf_counter() - t0; F1_0, FPR_0, _ = val_stats(ens(base, 'va')); full_rows = N*RB*FL['local_epochs']
    # ---------------- Part B: the paper's protocol ----------------
    I_all, _ = contributions(base, np.ones(N, bool), set(range(CLIENTS)), F1_0)
    order = sorted(I_all, key=I_all.get, reverse=True); terc = np.array_split(order, 3); rb = np.random.default_rng(123)
    targets = [(g, int(c)) for g, tt in zip(['high', 'medium', 'low'], terc) for c in rb.choice(tt, 3, replace=False)]
    for group, c in targets:
        alive = np.ones(N, bool); alive[parts[c]] = False; present = set(range(CLIENTS)) - {c}; j = shard_of[c]; idx = shard_idx(j, alive)
        res = {}
        t1 = time.perf_counter(); full = []                                     # Full Retrain: every shard from scratch
        for jj in range(S):
            t = Trainer(shard_idx(jj, alive), s + 1000*jj)
            for _ in range(RB): t.step()
            full.append(make(t))
        res['full_retrain'] = (full, time.perf_counter() - t1, S*0 + sum(len(shard_idx(jj, alive)) for jj in range(S))*RB*FL['local_epochs'], RB)
        res['none'] = (base, 0., 0, 0)
        for name, R in [('affected_shard', RB), ('rapid', FIXED_R)]:
            t1 = time.perf_counter(); t = Trainer(idx, s + 1000*j)
            for _ in range(R): t.step()
            sh = list(base); sh[j] = make(t); res[name] = (sh, time.perf_counter() - t1, t.rows*R*FL['local_epochs'], R)
        t1 = time.perf_counter(); e, t, info = dynamicfu(base, j, idx, s + 1000*j, I_all[c], F1_0); sh = list(base); sh[j] = e
        res['dynamicfu'] = (sh, time.perf_counter() - t1, t.rows*t.r*FL['local_epochs'], t.r)
        t1 = time.perf_counter(); e, t = utility_stop(base, j, idx, s + 1000*j, F1_0, FPR_0); sh = list(base); sh[j] = e
        res['utility_stop'] = (sh, time.perf_counter() - t1, t.rows*t.r*FL['local_epochs'], t.r)
        ref = report(res['full_retrain'][0], alive, present)
        for name, (sh, secs, rws, R) in res.items():
            rep = report(sh, alive, present); exact = name != 'none' and all(not (~alive[e['idx']]).any() for e in sh if e is not None and e['in'])
            rowsB.append({'alpha': ALPHA, 'seed': s, 'group': group, 'client': c, 'client_rows': len(parts[c]), 'contribution': I_all[c],
                          'method': name, 'rounds': R, 'seconds': secs, 'rows_x_epochs': rws, 'T_full': T_full, 'full_rows_x_epochs': full_rows,
                          'exact': exact, **rep, 'dAcc': abs(rep['cal_acc'] - ref['cal_acc']), 'dF1': abs(rep['cal_macro_f1'] - ref['cal_macro_f1']),
                          'dFPR': abs(rep['cal_fpr'] - ref['cal_fpr']), **(info if name == 'dynamicfu' else {})})
        same = all(all(torch.equal(a, b) for a, b in zip(x['m'].state_dict().values(), y['m'].state_dict().values()))
                   for x, y in zip(res['full_retrain'][0], res['affected_shard'][0]))
        checks.append({'alpha': ALPHA, 'seed': s, 'part': 'B', 'client': c, 'affected_shard_equals_full_retrain': same})
    # ---------------- Part A: sequential stream ----------------
    for pol in ['none', 'shard_full', 'shard_fixed', 'dynamicfu', 'utility_stop', 'drop_backfill']:
        sh = list(base); alive = np.ones(N, bool); present = set(range(CLIENTS)); pending = []
        cost = dict(rows_x_epochs=0, contrib_rows=0, rounds=0, bytes=0, seconds=0.)
        def charge(t, r0, t0):
            n = t.r - r0; cost['rows_x_epochs'] += t.rows*n*FL['local_epochs']; cost['rounds'] += n
            cost['bytes'] += 2*MB*t.nclients*n; cost['seconds'] += time.perf_counter() - t0
        for req in stream:
            alive_before = alive.copy(); alive[req['f']] = False
            if req['kind'] == 'client': present.discard(req['client'])
            j = shard_of[req['client']]; idx = shard_idx(j, alive); t0 = time.perf_counter(); extra = {}
            if pol != 'none' and not len(idx): sh[j] = None; pending = [p for p in pending if p[0] != j]
            elif pol in ('shard_full', 'shard_fixed'):
                t = Trainer(idx, s + 1000*j)
                for _ in range(RB if pol == 'shard_full' else FIXED_R): t.step()
                sh[j] = make(t); charge(t, 0, t0)
            elif pol == 'utility_stop':
                sh[j], t = utility_stop(sh, j, idx, s + 1000*j, F1_0, FPR_0); charge(t, 0, t0)
            elif pol == 'dynamicfu':
                F1_now = val_stats(ens(sh, 'va'))[0]; I, crow = contributions(sh, alive_before, present | {req['client']}, F1_now)
                cost['contrib_rows'] += crow*FL['local_epochs']
                sh[j], t, extra = dynamicfu(sh, j, idx, s + 1000*j, I.get(req['client'], 0.), F1_now); charge(t, 0, t0)
            elif pol == 'drop_backfill':
                if sh[j] is not None: sh[j] = {**sh[j], 'in': False}
                pending = [p for p in pending if p[0] != j] + [(j, Trainer(idx, s + 1000*j))]; budget = BACKFILL
                while budget and pending:
                    jj, t = pending[0]; r0 = t.r; t1 = time.perf_counter(); n = min(budget, RB - t.r)
                    for _ in range(n): t.step()
                    charge(t, r0, t1); budget -= n
                    if t.r >= RB: sh[jj] = make(t); pending.pop(0)
                extra['pending'] = len(pending)
            exact = all(not (~alive[e['idx']]).any() for e in sh if e is not None and e['in'])
            rowsA.append({'alpha': ALPHA, 'seed': s, 'policy': pol, 'step': req['step'], 'kind': req['kind'], 'exact': exact, **cost,
                          **report(sh, alive, present), **extra})
        if pol == 'drop_backfill':
            for jj, t in pending:
                r0 = t.r; t1 = time.perf_counter()
                while t.r < RB: t.step()
                charge(t, r0, t1); sh[jj] = make(t)
            rowsA.append({'alpha': ALPHA, 'seed': s, 'policy': pol, 'step': K + 1, 'kind': 'backlog_done', 'exact': True, **cost, **report(sh, alive, present)})
        if pol != 'none':
            ok, n = bit_exact(sh, alive); checks.append({'alpha': ALPHA, 'seed': s, 'part': 'A', 'policy': pol, 'shards': n, 'bit_exact': ok})
    # reference: single model retrained from scratch after every request
    alive = np.ones(N, bool); present = set(range(CLIENTS)); cum = dict(rows_x_epochs=0, rounds=0, bytes=0, seconds=0.)
    for req in stream:
        alive[req['f']] = False
        if req['kind'] == 'client': present.discard(req['client'])
        t0 = time.perf_counter(); t = Trainer(np.flatnonzero(alive), s)
        for _ in range(RB): t.step()
        cum['rows_x_epochs'] += t.rows*RB*FL['local_epochs']; cum['rounds'] += RB; cum['bytes'] += 2*MB*t.nclients*RB; cum['seconds'] += time.perf_counter() - t0
        rowsA.append({'alpha': ALPHA, 'seed': s, 'policy': 'full_retrain_single', 'step': req['step'], 'kind': req['kind'], 'exact': True, **cum,
                      **report([make(t)], alive, present)})
    pd.DataFrame(rowsA).to_csv(OUT/'stream_steps.csv', index=False); pd.DataFrame(rowsB).to_csv(OUT/'paper_cases.csv', index=False)
    pd.DataFrame(checks).to_csv(OUT/'checks.csv', index=False)
    print(f'alpha={ALPHA} seed={s} done', flush=True)
print('DONE', flush=True)
