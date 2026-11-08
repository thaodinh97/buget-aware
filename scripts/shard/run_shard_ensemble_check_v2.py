"""Run 1 of the v2 shard protocol (docs/shard_v2_plan.md): utility of sharded FL ensembles and of exact unlearning of one
request by retraining only the affected shard. Usage: python run_shard_ensemble_check_v2.py OUT_DIR

Fixed before running (design from scripts/shard/shard_v2.py, SHARD_DESIGN, default 'v2'):
- data: SHARD_DATA_DIR (default connected_v3_auxfit_seed42), content hash checked; metrics on CALIBRATION; test unused;
- 20 clients, Dirichlet alpha 0.5 on labels, partition seed 42 (first valid seed >= 42);
- S in {1, 2, 4, 5}: shards by seeded permutation (seed 7), every shard >= 4 clients (checked);
- each shard: FedAvg with the 'standard' config (R_base 20 rounds, 2 local epochs, Adam 1e-2, batch 512); shard j seed s+1000j;
  workers are numbered by client id (as in the stream runner), so shard models equal the stream runner's base shards;
- ensemble prediction: sum of shard logit margins, threshold 0.5;
- training seeds 50042..50051 (never run before);
- requests on the largest client t: 'client' = delete all of t; 'random50' = stratified random 50% of t (seed 43);
  the affected shard must keep >= merge_below (3) clients (checked; no merge is needed in run 1);
- unlearning: affected shard re-initialised with its seed and retrained on its retained data; utility after R in
  {1, 2, 5, 10, 20} rounds (prefix of one run; R = R_base is the exact retrain of the ensemble with the same seeds).
"""
import json, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from scipy.special import expit
from src.data.federated import partition, forget_request
from src.eval.membership_validation import logit_margin
from src.eval.utility_breakdown import detector_metrics
from src.fl.fedavg_core import model, fedavg, ClientWorker
from scripts.shard.shard_v2 import start_run, assign, check_design

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
DESIGN_NAME, DESIGN, CFG, d = start_run(OUT, __file__)
FL = CFG['fl']; HIDDEN = [64, 32]
CLIENTS, ALPHA, SHARDS, SEEDS, BUDGETS = 20, .5, DESIGN['shards_run1'], DESIGN['seeds'], CFG['unlearn_budgets']
tr, ca = d['train'], d['calibration']; DIM = tr['X'].shape[1]

pseed = 42
while True:
    try: parts = partition(tr['y'], pseed, ALPHA, CLIENTS); break
    except ValueError: pseed += 1
sizes = [len(p) for p in parts]; t = int(np.argmax(sizes))
f_rand, _ = forget_request(tr['y'], parts[t], .5, 43)
requests = {'client': parts[t], 'random50': f_rand}
shards_of = {S: assign(S, CLIENTS) for S in SHARDS}
for S, A in shards_of.items():
    check_design(A, DESIGN['min_clients_design'])
    left = len(next(a for a in A if t in a)) - 1
    if S > 1 and left < DESIGN['merge_below']: raise ValueError(f'S={S}: deleting client {t} leaves {left} clients in its shard')
meta = {'design': DESIGN_NAME, 'partition_seed': pseed, 'target_client': t, 'client_rows': sizes,
        'client_normal': [int((tr['y'][p] == 0).sum()) for p in parts], 'request_rows': {k: len(v) for k, v in requests.items()},
        'shards': shards_of}
print(json.dumps({k: v for k, v in meta.items() if k != 'shards'}), flush=True)
model_bytes = sum(p.numel()*4 for p in model(DIM, HIDDEN, 0, 'cpu').parameters())

def train_rounds(clients, blocks, seed, rounds):
    """Yield (r, model, rows, workers) after each round r = 1..rounds; equal to fedavg(..., rounds, seed) at every prefix."""
    ws = [ClientWorker(c, tr['X'][b], tr['y'][b], tr['row_id'][b], 'cpu') for c, b in zip(clients, blocks) if len(b)]
    m = model(DIM, HIDDEN, seed, 'cpu')
    for r in range(rounds):
        m, _ = fedavg(m, ws, FL, 1, seed + r*10000); yield r + 1, m, sum(len(w.y) for w in ws), len(ws)

def score(margins, extra):
    cal = sum(m[0] for m in margins); out = {**extra, **{f'cal_{k}': v for k, v in detector_metrics(ca['y'], expit(cal)).items()}}
    cl = [detector_metrics(tr['y'][p], expit(sum(m[1][i] for m in margins))) for i, p in enumerate(parts)]
    f1 = np.array([c['macro_f1'] for c in cl]); out.update(client_f1_mean=float(np.nanmean(f1)), client_f1_min=float(np.nanmin(f1)),
        client_f1_min_id=int(np.nanargmin(f1)), **{f'client{i}_f1': float(v) for i, v in enumerate(f1)})
    return out

def margins_of(m): return (logit_margin(m, ca['X']), [logit_margin(m, tr['X'][p]) for p in parts])

rows, unl = [], []
for S in SHARDS:
    shards = shards_of[S]; ts = next(j for j, sh in enumerate(shards) if t in sh)
    for s in SEEDS:
        t0 = time.perf_counter(); base = []
        for j, sh in enumerate(shards):
            for r, m, n, k in train_rounds(sh, [parts[i] for i in sh], s + 1000*j, FL['rounds']): pass
            base.append(margins_of(m))
        secs = time.perf_counter() - t0
        rows.append(score(base, {'S': S, 'seed': s, 'train_seconds': secs, 'rows_x_rounds': len(tr['y'])*FL['rounds']}))
        for req, f in requests.items():
            keep = [np.setdiff1d(parts[i], f) for i in shards[ts]]
            t0 = time.perf_counter()
            for r, m, n, k in train_rounds(shards[ts], keep, s + 1000*ts, FL['rounds']):
                if r in BUDGETS:
                    el = time.perf_counter() - t0
                    mg = [margins_of(m) if j == ts else base[j] for j in range(S)]
                    unl.append(score(mg, {'S': S, 'seed': s, 'request': req, 'rounds': r, 'retrain_seconds': el,
                        'rows_x_rounds': n*r, 'clients_in_shard': k, 'bytes': 2*model_bytes*k*r,
                        'full_retrain_rows_x_rounds': (len(tr['y']) - len(f))*FL['rounds'],
                        'full_retrain_bytes': 2*model_bytes*(CLIENTS - (req == 'client'))*FL['rounds']}))
        print(f'S={S} seed={s} done ({secs:.1f}s train)', flush=True)
pd.DataFrame(rows).to_csv(OUT/'ensemble.csv', index=False); pd.DataFrame(unl).to_csv(OUT/'unlearning.csv', index=False)
(OUT/'meta.json').write_text(json.dumps(meta, indent=1))
print('DONE', flush=True)
