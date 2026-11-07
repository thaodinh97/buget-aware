"""Feasibility check (exploratory, not preregistered; results in results/shard_exploration): utility cost of sharded FL ensembles and
the cost/utility curve of exact unlearning by retraining only the affected shard (DynamicFU / SISA style).

Fixed before running:
- data: data/processed/connected_v3_seed42; metrics on the CALIBRATION split (test is never touched);
- 20 clients, Dirichlet alpha 0.5 on labels; partition seed = first seed >= 42 that gives a valid partition;
- S in {1,2,4,5,10} shards; clients assigned to shards by a seeded permutation (seed 7), equal client counts;
- each shard: FedAvg with the repo config (15 rounds, 1 local epoch, Adam 1e-3, batch 512), shard j seed = s + 1000*j;
- ensemble prediction: sum of shard logit margins (DynamicFU eq. 5), threshold 0.5;
- training seeds s = 10042..10046;
- requests on the largest client t: 'client' = delete all of t; 'random50' = stratified random 50% of t (seed 43);
- unlearning: affected shard re-initialised with its original seed and retrained on its retained data; utility after
  R in {1,2,5,10,15} rounds (prefix of one run; R = 15 is the exact retrain of the ensemble with the same seeds).
"""
import json, os, sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, pandas as pd, torch
from scipy.special import expit
from scripts.run_pilot import load_data
from src.data.prepare import digest
from src.data.federated import partition, forget_request
from src.eval.membership_validation import logit_margin
from src.eval.utility_breakdown import detector_metrics
from src.fl.fedavg_core import model, fedavg, ClientWorker

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=False)
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
FL = {'rounds': 15, 'local_epochs': 1, 'batch_size': 512, 'lr': 1e-3}; HIDDEN = [64, 32]
CLIENTS, ALPHA, SHARDS, SEEDS, BUDGETS = 20, .5, [1, 2, 4, 5, 10], [10042 + i for i in range(5)], [1, 2, 5, 10, 15]
DATA = Path(os.environ.get('SHARD_DATA_DIR', ROOT/'data/processed/connected_v3_seed42'))  # default: the frozen train-fit data
d = load_data(DATA); tr, ca = d['train'], d['calibration']
DIM = d['train']['X'].shape[1]
(OUT/'data.json').write_text(json.dumps({'data_dir': str(DATA), 'input_dim': DIM, 'manifest_sha256': digest(Path(DATA)/'manifest.json')}, indent=1))

pseed = 42
while True:
    try: parts = partition(tr['y'], pseed, ALPHA, CLIENTS); break
    except ValueError: pseed += 1
sizes = [len(p) for p in parts]; t = int(np.argmax(sizes))
f_rand, _ = forget_request(tr['y'], parts[t], .5, 43)
requests = {'client': parts[t], 'random50': f_rand}
meta = {'partition_seed': pseed, 'target_client': t, 'client_rows': sizes,
        'client_normal': [int((tr['y'][p] == 0).sum()) for p in parts], 'request_rows': {k: len(v) for k, v in requests.items()}}
print(json.dumps(meta), flush=True)
model_bytes = sum(p.numel()*4 for p in model(DIM, HIDDEN, 0, 'cpu').parameters())

def assign(S):
    perm = np.random.default_rng(7).permutation(CLIENTS); return [sorted(perm[j::S].tolist()) for j in range(S)]

def train_rounds(blocks, seed, rounds):
    """Yield (r, model) after each round r = 1..rounds; identical to fedavg(..., rounds, seed) at every prefix."""
    ws = [ClientWorker(i, tr['X'][b], tr['y'][b], tr['row_id'][b], 'cpu') for i, b in enumerate(blocks) if len(b)]
    m = model(DIM, HIDDEN, seed, 'cpu')
    for r in range(rounds):
        m, _ = fedavg(m, ws, FL, 1, seed + r*10000); yield r + 1, m, sum(len(w.y) for w in ws), len(ws)

def score(margins, extra):
    """margins: list of per-shard (calibration margin, per-client margin list)."""
    cal = sum(m[0] for m in margins); out = {**extra, **{f'cal_{k}': v for k, v in detector_metrics(ca['y'], expit(cal)).items()}}
    cl = [detector_metrics(tr['y'][p], expit(sum(m[1][i] for m in margins))) for i, p in enumerate(parts)]
    f1 = np.array([c['macro_f1'] for c in cl]); out.update(client_f1_mean=float(np.nanmean(f1)), client_f1_min=float(np.nanmin(f1)),
        client_f1_min_id=int(np.nanargmin(f1)), **{f'client{i}_f1': float(v) for i, v in enumerate(f1)})
    return out

def margins_of(m): return (logit_margin(m, ca['X']), [logit_margin(m, tr['X'][p]) for p in parts])

rows, unl = [], []
for S in SHARDS:
    shards = assign(S); ts = next(j for j, sh in enumerate(shards) if t in sh)
    for s in SEEDS:
        t0 = time.perf_counter(); base = []
        for j, sh in enumerate(shards):
            for r, m, n, k in train_rounds([parts[i] for i in sh], s + 1000*j, FL['rounds']): pass
            base.append(margins_of(m))
        secs = time.perf_counter() - t0
        rows.append(score(base, {'S': S, 'seed': s, 'train_seconds': secs, 'rows_x_rounds': len(tr['y'])*FL['rounds']}))
        for req, f in requests.items():
            keep = [np.setdiff1d(parts[i], f) for i in shards[ts]]
            t0 = time.perf_counter()
            for r, m, n, k in train_rounds(keep, s + 1000*ts, FL['rounds']):
                if r in BUDGETS:
                    el = time.perf_counter() - t0
                    mg = [margins_of(m) if j == ts else base[j] for j in range(S)]
                    unl.append(score(mg, {'S': S, 'seed': s, 'request': req, 'rounds': r, 'retrain_seconds': el,
                        'rows_x_rounds': n*r, 'clients_in_shard': k, 'bytes': 2*model_bytes*k*r,
                        'full_retrain_rows_x_rounds': (len(tr['y']) - len(f))*FL['rounds'],
                        'full_retrain_bytes': 2*model_bytes*(CLIENTS - (req == 'client'))*FL['rounds']}))
        print(f'S={S} seed={s} done ({secs:.1f}s train)', flush=True)
pd.DataFrame(rows).to_csv(OUT/'ensemble.csv', index=False); pd.DataFrame(unl).to_csv(OUT/'unlearning.csv', index=False)
(OUT/'meta.json').write_text(json.dumps({**meta, 'shards': {S: assign(S) for S in SHARDS}}, indent=1))
print('DONE', flush=True)
