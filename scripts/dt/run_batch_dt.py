"""Deadline-aware batching of the v3 request stream (direction 3 of docs/dt_extensions.md).

  python scripts/dt/run_batch_dt.py OUT --seed 50042 --family concentrated

One process = one (S, training seed, stream family). The ensemble is trained once, without stress. Every mode then runs the
20 sample-level requests of the family (scripts/shard/stream_v3.py) on a fork of it through src/dt/scheduler.run; every
processing event is one ShardedUnlearner.unlearn_batch call, so a shard is retrained once per event however many requests
of the event touch it.

Modes (labels): immediate, window5 (the v3 policy batch_full_5), lazy, dt_mid, dt_tight. The dt modes use the Adaptive twin
with the q90 error margin (src/dt/planner.py) and the budget level in their label (x the measured fit time of the largest
shard, as in directions 1-2); the other modes always use shard_full, and their events are scored against the mid budget.
Deadline D = 5 simulated time units (the largest latency of batch_full_5 is 4.95).

Writes to OUT (must not exist): run.json; <mode>/events.jsonl (one processing event per line: time, shards, requests,
latencies, policy, forecast, pooled budget, measured seconds, rows x rounds, validation metrics after the event, exactness);
<mode>/final.json (end state: validation metrics, verify_exact, SHA-256 of every shard's weights, and with --verify the
bit-for-bit retrain check); summary.csv (one row per mode). Validation only; calibration and test unused.
"""
import argparse
import hashlib
import json
import os
import platform
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from scripts.dt.machine import prefer_performance
from scripts.dt.run_stress_trajectories import stream_requests
from scripts.shard.shard_v2 import DATA_DEFAULT, DATA_SHA256, data_content_sha256
from scripts.shard.stream_v3 import FAMILIES, STREAM
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.dt import planner, scheduler
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner, assign_shards, _py

DEADLINE = 5.
MODES = {'immediate': ('immediate', None, 'mid'), 'window5': ('window', 5., 'mid'), 'lazy': ('lazy', None, 'mid'),
         'dt_mid': ('dt', None, 'mid'), 'dt_tight': ('dt', None, 'tight')}


def weights_sha(fed):
    out = {}
    for j, e in enumerate(fed.shards):
        if e is None:
            continue
        h = hashlib.sha256()
        for key, value in e.model.state_dict().items():
            h.update(key.encode()); h.update(value.detach().cpu().contiguous().numpy().tobytes())
        out[str(j)] = h.hexdigest()
    return out


def run_mode(base, label, requests, budget, fit_profile, fit_history, validation, verify):
    mode, window, _ = MODES[label]
    fed = base.fork('shard_full')

    def execute(batch, policy):
        result = fed.unlearn_batch([ForgetRequest(r['client'], r['f']) for r in batch], policy)
        met = detector_metrics(validation['y'], fed.predict_proba(validation['X']))
        return {'seconds': float(sum(x['wall_seconds'] for x in result.round_costs)), 'rows_x_rounds': result.cost['rows_x_rounds'],
                'rounds': result.cost['rounds'], 'shards_retrained': result.details['shards_retrained'],
                'exact': result.exact, 'structure_ok': result.structure_ok,
                'validation_macro_f1': met['macro_f1'], 'validation_fpr': met['fpr'],
                'round_costs': result.round_costs, 'client_costs': result.client_costs}

    def counts(shard, queued):
        out = {int(c): int(fed.alive[fed.parts[c]].sum()) for c in fed.assignment[shard]}
        for r in queued:
            out[r['client']] -= r['rows']
        return out

    twin = scheduler.DtTwin(fit_profile, fit_history[0], counts) if mode == 'dt' else None
    events = scheduler.run(requests, mode, DEADLINE, budget, execute, window, twin,
                           (base.config.fl['rounds'], base.config.shard_fixed_rounds))
    for e in events:
        print(json.dumps({'mode': label, 'time': round(e['time'], 3), 'shards': e['shards'], 'requests': e['requests'],
                          'policy': e['policy'], 'seconds': round(e['seconds'], 3)}), flush=True)
    met = detector_metrics(validation['y'], fed.predict_proba(validation['X']))
    final = {'validation_macro_f1': met['macro_f1'], 'validation_fpr': met['fpr'], **fed.verify_exact(),
             'weights_sha256': weights_sha(fed), 'shard_rounds': {str(j): e.rounds for j, e in enumerate(fed.shards) if e},
             'alive_rows': int(fed.alive.sum())}
    if verify:
        final['verify_bit_exact'] = fed.verify_bit_exact()
    return events, final


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--modes', nargs='+', choices=list(MODES), default=list(MODES))
    parser.add_argument('--verify', action='store_true', help='retrain every included shard at the end of every mode')
    parser.add_argument('--data', type=Path, default=ROOT/DATA_DEFAULT)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--train-config', default='standard', help=argparse.SUPPRESS)   # tests only
    args = parser.parse_args(argv)
    from src.data.loading import load_data
    args.output.mkdir(parents=True, exist_ok=False)
    process = prefer_performance()
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    data = load_data(args.data); content = data_content_sha256(data)
    parts = partition(data['train']['y'], 42, .5, 20)
    config = ShardConfig.from_train_config(CONFIGS[args.train_config], min_clients_design=4, merge_below=3)
    base = ShardedUnlearner(data['train'], data['validation'], parts, assign_shards(args.shards, 20), args.seed, 'shard_full', config)
    fit = base.fit()
    fit_rounds, fit_clients = fit['round_costs'], fit['client_costs']
    profile = planner.rates(fit_rounds, fit_clients)
    history = planner.fit_ratios(fit_rounds, fit_clients, profile, config.shard_fixed_rounds)
    per_shard = pd.DataFrame(fit_rounds).groupby('shard').wall_seconds.sum()
    largest = float(per_shard.max())
    budgets = {level: scale*largest for level, scale in planner.BUDGET_LEVELS.items()}
    requests = stream_requests(base, args.family)
    record = {'seed': args.seed, 'family': args.family, 'shards': args.shards, 'modes': args.modes, 'deadline': DEADLINE,
              'train_config': args.train_config, 'config': config.to_json(), 'data_content_sha256': content,
              'data_is_v2': content == DATA_SHA256, 'stream': STREAM, 'budgets': budgets, 'budget_levels': planner.BUDGET_LEVELS,
              'fit': {'reference': fit['reference'], 'shard_seconds': per_shard.to_dict(), 'largest_shard_full_seconds': largest,
                      'profile': profile, 'initial_ratios': history},
              'assignment': base.assignment_initial, 'requests': [{k: v for k, v in r.items() if k != 'f'} for r in requests],
              'machine': {'platform': platform.platform(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(),
                          'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                          'threads': args.threads, 'device': config.device, 'process': process}}
    (args.output/'run.json').write_text(json.dumps(_py(record), indent=1), encoding='utf-8')
    summary = []
    for label in args.modes:
        level = MODES[label][2]
        events, final = run_mode(base, label, requests, budgets[level], profile, history, data['validation'], args.verify)
        (args.output/label).mkdir()
        (args.output/label/'events.jsonl').write_text(''.join(json.dumps(_py(e)) + '\n' for e in events), encoding='utf-8')
        (args.output/label/'final.json').write_text(json.dumps(_py(final), indent=1), encoding='utf-8')
        summary.append({'mode': label, 'budget_level': level, 'budget_seconds': budgets[level],
                        **scheduler.summarize(events, DEADLINE),
                        'validation_f1_min': min(e['validation_macro_f1'] for e in events),
                        'validation_f1_final': final['validation_macro_f1'], 'validation_fpr_final': final['validation_fpr'],
                        'exact_all_events': all(e['exact'] and e['structure_ok'] for e in events),
                        'final_exact': final['exact'] and final['structure_ok']})
        pd.DataFrame(summary).to_csv(args.output/'summary.csv', index=False, float_format='%.6g')


if __name__ == '__main__':
    main()
