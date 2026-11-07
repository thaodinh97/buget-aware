"""Diagnostic (exploratory, post hoc; output in results/shard_exploration_auxfit): does the high-FPR mode on auxfit data come from under-training? Same 20-client partition as
run_shard_ensemble_check (Dirichlet 0.5, seed 42, shards by permutation seed 7), seed 10042, calibration metrics."""
import sys, time
from pathlib import Path
ROOT = Path(__file__).resolve().parents[2]; sys.path.insert(0, str(ROOT))
import numpy as np, torch
from scipy.special import expit
from scripts.run_pilot import load_data
from src.data.federated import partition
from src.eval.membership_validation import logit_margin
from src.eval.utility_breakdown import detector_metrics
from src.fl.fedavg_core import model, fedavg, ClientWorker
torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
CONFIGS = [('repo: lr1e-3 15r 1ep', 1e-3, 15, 1), ('lr1e-3 45r 1ep', 1e-3, 45, 1), ('lr1e-2 15r 1ep', 1e-2, 15, 1), ('paper: lr1e-2 20r 2ep', 1e-2, 20, 2)]
for data in ['connected_v3_seed42', 'connected_v3_auxfit_seed42']:
    d = load_data(ROOT/'data/processed'/data); tr, ca = d['train'], d['calibration']; dim = tr['X'].shape[1]
    parts = partition(tr['y'], 42, .5, 20); perm = np.random.default_rng(7).permutation(20)
    for S in [1, 4]:
        shards = [sorted(perm[j::S].tolist()) for j in range(S)]
        for name, lr, rounds, ep in CONFIGS:
            fl = {'rounds': rounds, 'local_epochs': ep, 'batch_size': 512, 'lr': lr}; t0 = time.perf_counter(); mg = 0
            for j, sh in enumerate(shards):
                ws = [ClientWorker(i, tr['X'][parts[i]], tr['y'][parts[i]], tr['row_id'][parts[i]], 'cpu') for i in sh]
                m, _ = fedavg(model(dim, [64, 32], 10042 + 1000*j, 'cpu'), ws, fl, rounds, 10042 + 1000*j); mg = mg + logit_margin(m, ca['X'])
            met = detector_metrics(ca['y'], expit(mg))
            print(f'{data:28s} S={S:2d} {name:22s} macro-F1 {met["macro_f1"]:.3f}  FPR {met["fpr"]:.3f}  ({time.perf_counter()-t0:.0f}s)', flush=True)
