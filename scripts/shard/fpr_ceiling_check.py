"""Diagnostics D1-D3 of docs/fpr_threshold_plan.md (addendum 2): is the FPR limit in the model or in the features?

    FL_DEVICE=cpu python scripts/shard/fpr_ceiling_check.py OUT [--data-dir DIR --partition-seed N]

- fl_mlp_zscore   S = 1, the existing encoding (D1: adds train-set metrics; calibration equals results/fpr_threshold)
- fl_mlp_log1p    S in {1, 4, 5}, numeric columns log1p-transformed and re-standardized on validation (D3)
- gbdt_central    sklearn HistGradientBoostingClassifier, default parameters, trained centrally on train (D2;
                  a reference only: no FL, no sharded unlearning)
Same seeds 50042..50046 and threshold rules (chosen on validation) as fpr_threshold_check.py. Test is never opened.
Writes OUT (must not exist): run.json, thresholds.csv, curves.csv.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.shard.fpr_threshold_check import RULES, SEEDS, SHARDS, choose, fpr_at_recall, sweep
from scripts.shard.shard_v2 import DATA_DEFAULT, DATA_SHA256, assign, data_content_sha256, text_sha256
from scripts.shard.shard_v3 import EXPECTED_CLIENT_ROWS, load_protocol_data, machine_info, write_json
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import ShardConfig, ShardedUnlearner, warm_up

TRACKED = ['docs/fpr_threshold_plan.md', 'scripts/shard/fpr_threshold_check.py', 'scripts/shard/fpr_ceiling_check.py']


def log1p_variant(data, state):
    """Recover the numeric values from the z-scores, log1p them, re-standardize on validation; categorical columns kept."""
    num = state['numeric']
    k = len(num)
    if list(state['feature_names'][:k]) != list(num):
        raise ValueError('numeric features are not the leading encoded columns')
    scale = np.array([state['scale'][c] for c in num])
    mean = np.array([state['mean'][c] for c in num])
    integer = np.array([c != 'duration' for c in num])

    def logs(X):
        raw = X[:, :k].astype(np.float64)*scale + mean
        raw[:, integer] = np.round(raw[:, integer])
        return np.log1p(np.maximum(raw, 0))

    values = {name: logs(split['X']) for name, split in data.items()}
    mu, sd = values['validation'].mean(0), values['validation'].std(0)
    sd[sd == 0] = 1
    out = {}
    for name, split in data.items():
        X = split['X'].copy()
        X[:, :k] = ((values[name] - mu)/sd).astype(np.float32)
        out[name] = {**split, 'X': X}
    return out


def evaluate(base, data, score, with_train):
    """Threshold rules on validation scores; metrics on validation, calibration (and train); threshold-free curve."""
    m = {name: score(data[name]['X']) for name in (('train',) if with_train else ()) + ('validation', 'calibration')}
    s_va = sweep(m['validation'], data['validation']['y'])
    rows = []
    for rule in RULES:
        t = choose(rule, s_va)
        row = {**base, 'rule': rule, 'threshold': t}
        for name, tag in (('validation', 'val'), ('calibration', 'cal'), ('train', 'train')):
            if name in m:
                got = detector_metrics(data[name]['y'], m[name], t)
                row.update({f'{tag}_{k}': got[k] for k in ('macro_f1', 'fpr', 'recall_attack')})
        rows.append(row)
    y_ca = data['calibration']['y']
    s_ca = sweep(m['calibration'], y_ca)
    curve = {**base, 'cal_roc_auc': roc_auc_score(y_ca, m['calibration']),
             'cal_avg_precision': average_precision_score(y_ca, m['calibration']),
             'cal_fpr_at_recall_95': fpr_at_recall(s_ca, .95), 'cal_fpr_at_recall_99': fpr_at_recall(s_ca, .99)}
    return rows, curve


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--data-dir', default=DATA_DEFAULT)
    parser.add_argument('--partition-seed', type=int, default=42)
    args = parser.parse_args(argv)
    if args.out.exists():
        raise FileExistsError(f'Refusing overwrite: {args.out}')
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frozen = Path(args.data_dir) == Path(DATA_DEFAULT)
    if frozen and args.partition_seed != 42:
        raise ValueError('the frozen data uses partition seed 42')
    data, manifest = load_protocol_data(ROOT/args.data_dir)
    content = data_content_sha256(data)
    if frozen and content != DATA_SHA256:
        raise ValueError(f'data content {content} != expected {DATA_SHA256}')
    parts = partition(data['train']['y'], args.partition_seed, .5, 20)
    if frozen and [len(p) for p in parts] != EXPECTED_CLIENT_ROWS:
        raise ValueError('client row counts differ from the frozen partition')
    state = json.loads((ROOT/args.data_dir/'transformer.json').read_text(encoding='utf-8'))
    variants = {'zscore': data, 'log1p': log1p_variant(data, state)}
    config = ShardConfig.from_train_config(CONFIGS['standard'])
    args.out.mkdir(parents=True)
    record = {'plan': 'docs/fpr_threshold_plan.md (addendum 2)', 'sha256': {f: text_sha256(ROOT/f) for f in TRACKED},
              'data_dir': Path(args.data_dir).as_posix(), 'frozen_data': frozen, 'partition_seed': args.partition_seed,
              'data_content_sha256': content, 'manifest_artifacts_sha256': manifest['artifacts_sha256'],
              'seeds': SEEDS, 'shards': SHARDS, 'rules': RULES, 'component_config': config.to_json(),
              'machine': machine_info('cpu'), 'completed': False}
    write_json(args.out/'run.json', record)
    record['backend_warmup_seconds'] = warm_up(data['train']['X'].shape[1], config)
    rows, curves = [], []

    def save():
        pd.DataFrame(rows).to_csv(args.out/'thresholds.csv', index=False)
        pd.DataFrame(curves).to_csv(args.out/'curves.csv', index=False)

    jobs = [('fl_mlp_zscore', 'zscore', 1)] + [('fl_mlp_log1p', 'log1p', S) for S in SHARDS] + [('gbdt_central', 'zscore', 0)]
    for model_name, variant, S in jobs:
        d = variants[variant]
        for seed in SEEDS:
            started = time.perf_counter()
            if model_name == 'gbdt_central':
                clf = HistGradientBoostingClassifier(random_state=seed).fit(d['train']['X'], d['train']['y'])
                score = clf.decision_function
            else:
                assignment = [list(range(20))] if S == 1 else assign(S, 20)
                fed = ShardedUnlearner(d['train'], d['validation'], parts, assignment, seed, 'shard_full', config,
                                       measure_payload=False)
                fed.fit()
                score = fed.margin
            fit_seconds = time.perf_counter() - started
            base = {'model': model_name, 'S': S, 'seed': seed, 'fit_seconds': fit_seconds}
            got, curve = evaluate(base, d, score, with_train=S in (0, 1))
            rows.extend(got)
            curves.append(curve)
            save()
            print(f'{model_name} S={S} seed={seed} {fit_seconds:.1f}s cal FPR default {got[0]["cal_fpr"]:.3f}', flush=True)
    record.update(completed=True)
    write_json(args.out/'run.json', record)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
