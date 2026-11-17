"""Measured cost trajectories of the v3 request stream under injected stress: the input of the Digital Twin experiments of
docs/dt_extensions.md (direction 1: tight budgets, stress, safety margin; direction 2: mid-course adjustment).

  python scripts/dt/run_stress_trajectories.py OUT --seed 50042 --family spread

One process = one (S, training seed, stream family). The ensemble is trained once without stress; every environment of
src/dt/stress.py then runs the 20 sample-level requests of the family (scripts/shard/stream_v3.py) on a fork of it, with every
request processed by shard_full and the stress injected through the component's worker_factory hook. The weights do not depend
on the stress (tests/test_dt_extensions.py), only the measured times do. The environments run interleaved, request by
request, in an order that rotates with the step, so that a drift of the machine's own speed hits all of them at the same
requests; the process opts out of Windows power throttling first (scripts/dt/machine.py).

Why shard_full only: the first shard_fixed_rounds rounds of shard_full are shard_fixed, bit for bit (same trainer, same
per-round seeds), so one run measures the true cost of both options of every request, in the same state: the cost of
shard_fixed is the wall time of rounds 1..10, the cost of shard_full that of rounds 1..20. A planner that would have chosen
shard_fixed, or stopped at the checkpoint, is evaluated on exactly the rounds it would have run. Rejected requests are
assumed to be escalated and executed outside the budget, so every planner sees the same sequence of states.

Writes to OUT (must not exist):
  run.json              scenario, machine, stress parameters, slow clients, fit summary, the largest shard's measured fit cost
  fit_round_costs.csv / fit_client_costs.csv   cost records of the unstressed base fit (the Static twin's calibration)
  <env>/requests.jsonl  per request: arrival time, client, shard, rows, the twin's client counts of the touched shard after the
                        deletion, the condition, every round's wall/server/compute seconds, every client-round's samples and
                        compute seconds, validation macro-F1 / FPR after the request, exact / structure_ok
Validation only (the split the policies may read); calibration and test unused.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from scripts.dt.machine import prefer_performance
from scripts.shard.shard_v2 import DATA_DEFAULT, DATA_SHA256, data_content_sha256
from scripts.shard.stream_v3 import FAMILIES, STREAM, plan_stream, select_rows
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.dt.stress import ENVIRONMENTS, STRESS, environment, install, slow_clients
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner, assign_shards, _py

ROUND_KEYS = ['shard', 'round', 'clients', 'samples', 'wall_seconds', 'compute_seconds_sum', 'compute_seconds_max', 'server_seconds']
CLIENT_KEYS = ['round', 'client_id', 'samples', 'compute_seconds']


def stream_requests(fed, family, stream_cfg=STREAM):
    """The family's requests with their rows: every request deletes rows chosen among the client's rows still alive after
    all earlier requests (stream_v3.select_rows), so any prefix of the stream is a valid sequence of deletions."""
    client_rows = [len(p) for p in fed.parts]
    alive = np.ones(len(fed.alive), bool)
    out = []
    for r in plan_stream(family, client_rows, fed.assignment_initial, stream_cfg):
        rows = select_rows(fed.parts[r['client']][alive[fed.parts[r['client']]]], r['rows'], r['step'], stream_cfg['seed_rows'])
        alive[rows] = False
        out.append({**r, 'f': rows})
    return out


def shard_counts(fed, shard, client, rows):
    """The twin's view of the touched shard: samples per client after the deletion (what scripts/shard/shard_dt.plan uses)."""
    return {int(c): int(fed.alive[fed.parts[c]].sum()) - (rows if c == client else 0) for c in fed.assignment[shard]}


def run_request(fed, controller, r, validation):
    controller.step = r['step']
    cond = controller.condition()
    counts = shard_counts(fed, r['shard'], r['client'], len(r['f']))
    started = time.perf_counter()
    result = fed.unlearn(ForgetRequest(r['client'], r['f']))
    wall = time.perf_counter() - started
    met = detector_metrics(validation['y'], fed.predict_proba(validation['X']))
    return _py({'step': r['step'], 'time': r['time'], 'client': r['client'], 'shard': r['shard'], 'rows': len(r['f']),
                'requested': r['requested'], 'retrained_shard': result.retrained_shard, 'counts': counts,
                'condition': {'slow': cond.slow, 'bandwidth_bps': cond.bandwidth_bps if np.isfinite(cond.bandwidth_bps) else None},
                'rounds': [{k: x[k] for k in ROUND_KEYS} for x in result.round_costs],
                'clients': [{k: x[k] for k in CLIENT_KEYS} for x in result.client_costs],
                'exact': result.exact, 'structure_ok': result.structure_ok, 'operation_wall_seconds': wall,
                'started_unix': time.time() - wall, 'validation_macro_f1': met['macro_f1'], 'validation_fpr': met['fpr']})


def run_interleaved(base, envs, requests, validation, out):
    """Every environment on its own fork, interleaved request by request (the order of the environments rotates with the
    step), so a drift of the machine's speed during the run hits every environment at the same requests."""
    forks = {}
    for env in envs:
        fed = base.fork('shard_full')
        forks[env] = (fed, install(fed, environment(env, fed.assignment_initial)))
        (out/env).mkdir()
    for r in requests:
        k = r['step'] % len(envs)
        for position, env in enumerate(envs[k:] + envs[:k]):
            row = run_request(*forks[env], r, validation)
            row['position_in_step'] = position
            with open(out/env/'requests.jsonl', 'a', encoding='utf-8') as handle:
                handle.write(json.dumps(row) + '\n')
            print(json.dumps({'env': env, 'step': r['step'], 'shard': r['shard'],
                              'seconds': round(sum(x['wall_seconds'] for x in row['rounds']), 3)}), flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--envs', nargs='+', choices=ENVIRONMENTS, default=list(ENVIRONMENTS))
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
    requests = stream_requests(base, args.family)
    fit_rounds = pd.DataFrame(fit['round_costs'])
    per_shard = fit_rounds.groupby('shard').wall_seconds.agg(full='sum')
    per_shard['fixed'] = fit_rounds[fit_rounds['round'] <= config.shard_fixed_rounds].groupby('shard').wall_seconds.sum()
    fit_rounds.to_csv(args.output/'fit_round_costs.csv', index=False, float_format='%.9g')
    pd.DataFrame(fit['client_costs']).to_csv(args.output/'fit_client_costs.csv', index=False, float_format='%.9g')
    record = {'seed': args.seed, 'family': args.family, 'shards': args.shards, 'envs': args.envs, 'train_config': args.train_config,
              'config': config.to_json(), 'data_content_sha256': content, 'data_is_v2': content == DATA_SHA256,
              'stream': STREAM, 'stress': STRESS, 'slow_clients': slow_clients(base.assignment_initial, STRESS['seed_slow']),
              'assignment': base.assignment_initial, 'model_bytes': base.model_bytes,
              'fit': {'reference': fit['reference'], 'shard_seconds': per_shard.to_dict('index'),
                      'largest_shard_full_seconds': float(per_shard.full.max()),
                      'largest_shard_fixed_seconds': float(per_shard.fixed[per_shard.full.idxmax()]),
                      'backend_warmup_seconds': fit['cost']['backend_warmup_seconds']},
              'requests': [{k: v for k, v in r.items() if k != 'f'} for r in requests],
              'machine': {'platform': platform.platform(), 'processor': platform.processor(), 'cpu_count': os.cpu_count(),
                          'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                          'threads': args.threads, 'device': config.device, 'process': process}}
    (args.output/'run.json').write_text(json.dumps(_py(record), indent=1), encoding='utf-8')
    run_interleaved(base, args.envs, requests, data['validation'], args.output)


if __name__ == '__main__':
    main()
