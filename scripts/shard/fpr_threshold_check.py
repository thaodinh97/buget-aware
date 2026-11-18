"""Exploratory FPR check: choose the decision threshold on validation, report on calibration (docs/fpr_threshold_plan.md).

    FL_DEVICE=cpu python scripts/shard/fpr_threshold_check.py results/fpr_threshold

Trains the initial ensemble (no deletion) for S in {1, 4, 5} and seeds 50042..50046 with the v2/v3 'standard' setup,
then applies the threshold rules fixed in the plan to the validation margins. The test split is never opened.
Writes to OUT (must not exist): run.json, thresholds.csv, curves.csv, bayes.json.
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
from sklearn.metrics import average_precision_score, roc_auc_score

from scripts.shard.shard_v2 import DATA_DEFAULT, DATA_SHA256, assign, data_content_sha256, text_sha256
from scripts.shard.shard_v3 import EXPECTED_CLIENT_ROWS, load_protocol_data, machine_info, write_json
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import ShardConfig, ShardedUnlearner, warm_up

SEEDS = list(range(50042, 50047))
SHARDS = [1, 4, 5]
RULES = ['default', 'max_f1', 'youden', 'fpr10', 'fpr05']
TRACKED = ['docs/fpr_threshold_plan.md', 'scripts/shard/fpr_threshold_check.py']


def sweep(margin, y):
    """Confusion counts for every distinct threshold t (predict attack when margin >= t), t ascending."""
    t, inverse = np.unique(margin, return_inverse=True)
    pos = np.bincount(inverse, weights=(y == 1), minlength=len(t))
    neg = np.bincount(inverse, weights=(y == 0), minlength=len(t))
    tp, fp = np.cumsum(pos[::-1])[::-1], np.cumsum(neg[::-1])[::-1]
    P, N = pos.sum(), neg.sum()
    fn, tn = P - tp, N - fp
    with np.errstate(invalid='ignore', divide='ignore'):
        f1_attack = np.where(2*tp + fp + fn > 0, 2*tp/(2*tp + fp + fn), 0.)
        f1_normal = np.where(2*tn + fn + fp > 0, 2*tn/(2*tn + fn + fp), 0.)
    return {'t': t, 'tpr': tp/P, 'fpr': fp/N, 'macro_f1': (f1_attack + f1_normal)/2}


def pick(t, score, allowed=None):
    """argmax of score (over allowed), ties broken by the smallest |t|."""
    score = np.where(np.ones_like(score, bool) if allowed is None else allowed, score, -np.inf)
    best = np.flatnonzero(score == score.max())
    return float(t[best[np.argmin(np.abs(t[best]))]])


def choose(rule, s):
    if rule == 'default':
        return 0.
    if rule == 'max_f1':
        return pick(s['t'], s['macro_f1'])
    if rule == 'youden':
        return pick(s['t'], s['tpr'] - s['fpr'])
    q = {'fpr10': .10, 'fpr05': .05}[rule]
    return pick(s['t'], s['tpr'], s['fpr'] <= q)


def fpr_at_recall(s, r):
    ok = np.flatnonzero(s['tpr'] >= r)
    return float(s['fpr'][ok[-1]])  # largest t that still reaches recall r


def bayes(split):
    """Sample Bayes error over identical feature vectors, and the normal rows a per-vector majority rule must flag."""
    _, inverse = np.unique(split['X'], axis=0, return_inverse=True)
    inverse = inverse.reshape(-1)
    n1 = np.bincount(inverse, weights=(split['y'] == 1))
    n0 = np.bincount(inverse, weights=(split['y'] == 0))
    conflict = (n0 > 0) & (n1 > 0)
    return {'rows': int(len(inverse)), 'distinct_vectors': int(len(n0)),
            'normal_rows': int(n0.sum()), 'vectors_with_both_labels': int(conflict.sum()),
            'rows_in_conflicting_vectors': int((n0 + n1)[conflict].sum()),
            'bayes_error': float(np.minimum(n0, n1).sum()/len(inverse)),
            'fpr_floor_majority': float(n0[n1 > n0].sum()/n0.sum())}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--data-dir', default=DATA_DEFAULT,
                        help='processed data; the frozen hash and client counts are enforced only for the default')
    parser.add_argument('--partition-seed', type=int, default=42,
                        help='Dirichlet partition seed (42 = the frozen partition; see the deviation note in the plan)')
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
    config = ShardConfig.from_train_config(CONFIGS['standard'])
    args.out.mkdir(parents=True)
    record = {'plan': 'docs/fpr_threshold_plan.md', 'sha256': {f: text_sha256(ROOT/f) for f in TRACKED},
              'data_dir': Path(args.data_dir).as_posix(), 'frozen_data': frozen, 'partition_seed': args.partition_seed,
              'manifest_artifacts_sha256': manifest['artifacts_sha256'], 'client_rows': [len(p) for p in parts],
              'data_content_sha256': content, 'seeds': SEEDS, 'shards': SHARDS, 'rules': RULES,
              'component_config': config.to_json(), 'machine': machine_info('cpu'), 'completed': False}
    write_json(args.out/'run.json', record)
    record['backend_warmup_seconds'] = warm_up(data['train']['X'].shape[1], config)
    va, ca = data['validation'], data['calibration']
    rows, curves = [], []
    for S in SHARDS:
        assignment = [list(range(20))] if S == 1 else assign(S, 20)
        for seed in SEEDS:
            fed = ShardedUnlearner(data['train'], va, parts, assignment, seed, 'shard_full', config, measure_payload=False)
            fit_started = time.perf_counter()
            fed.fit()
            fit_seconds = time.perf_counter() - fit_started
            started = time.perf_counter()
            m_va = fed.margin(va['X'])
            s_va = sweep(m_va, va['y'])
            thresholds = {rule: choose(rule, s_va) for rule in RULES}
            recalibration_seconds = time.perf_counter() - started
            m_ca = fed.margin(ca['X'])
            s_ca = sweep(m_ca, ca['y'])
            curves.append({'S': S, 'seed': seed, 'fit_seconds': fit_seconds, 'recalibration_seconds': recalibration_seconds,
                           'val_roc_auc': roc_auc_score(va['y'], m_va), 'cal_roc_auc': roc_auc_score(ca['y'], m_ca),
                           'cal_avg_precision': average_precision_score(ca['y'], m_ca),
                           'cal_fpr_at_recall_95': fpr_at_recall(s_ca, .95),
                           'cal_fpr_at_recall_99': fpr_at_recall(s_ca, .99)})
            for rule, t in thresholds.items():
                val, cal = detector_metrics(va['y'], m_va, t), detector_metrics(ca['y'], m_ca, t)
                rows.append({'S': S, 'seed': seed, 'rule': rule, 'threshold_margin': t,
                             **{f'val_{k}': val[k] for k in ('macro_f1', 'fpr', 'recall_attack')},
                             **{f'cal_{k}': cal[k] for k in ('macro_f1', 'fpr', 'recall_attack', 'f1_normal', 'f1_attack')}})
            print(f'S={S} seed={seed} fit {fit_seconds:.1f}s; cal FPR default {rows[-5]["cal_fpr"]:.3f}', flush=True)
            pd.DataFrame(rows).to_csv(args.out/'thresholds.csv', index=False)
            pd.DataFrame(curves).to_csv(args.out/'curves.csv', index=False)
    write_json(args.out/'bayes.json', {name: bayes(data[name]) for name in ('train', 'validation', 'calibration')})
    record.update(completed=True)
    write_json(args.out/'run.json', record)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
