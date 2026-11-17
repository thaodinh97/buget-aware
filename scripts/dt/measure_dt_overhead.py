"""The cost of the Digital Twin itself (docs/dt_extensions.md, "Chi phí của chính DT"). Usage:

  python scripts/dt/measure_dt_overhead.py results/dt_extensions/trajectories results/dt_extensions/overhead --live --seed 50042

Offline part (no training; ROOT holds the trajectories of scripts/dt/run_stress_trajectories.py). For every measured request:
  - the seconds the twin spends on plan and on sync (src/dt/overhead.time_twin, median of --repeats replays), for five twins:
    static, adaptive, adaptive + margin, adaptive + re-scored margin, last_shard;
  - the telemetry bytes the operation must report (records as JSON, the same compressed, and the sufficient-statistics
    summary), for the 20-round and the 10-round operation, against the raw model payload of the same operation;
  - both against the measured FU time of the request (round wall seconds).
Live part (--live: loads the data and trains the ensemble once, about one minute):
  - state: seconds to read the planner's view of one shard from the live unlearner (shard_counts), and seconds and JSON bytes
    of the full execution-state snapshot (client_state: 20 clients) and of the twin's stored profile;
  - metering: the first requests of the `spread` stream run twice from the same state, with the payload measurement of the
    CostMeter on (as in every experiment of this repository) and off, in alternating order. The difference is what measuring
    costs inside the simulation; the weights are identical either way.
The in-loop numbers (state + plan + sync around real operations) are recorded by scripts/dt/run_midcourse_online.py in every
row of its decisions.jsonl.

Writes OUT/twin_times.csv (per request and twin), OUT/telemetry.csv (per request), OUT/summary.json and, with --live,
OUT/live.json. Times are of this code on this machine; a byte count is a size, not a transfer time. Descriptive only.
"""
import argparse
import json
import sys
import time
from pathlib import Path
from statistics import median

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.dt.evaluate_stress import load_run
from src.dt import overhead, planner

TWINS = {'static': planner.PlannerConfig('static'), 'adaptive': planner.PlannerConfig('adaptive'),
         'adaptive_margin': planner.PlannerConfig('adaptive', .9),
         'adaptive_margin_rescored': planner.PlannerConfig('adaptive', .9, margin_mode='rescored'),
         'last_shard': planner.PlannerConfig('last_shard')}


def offline(root, repeats):
    times, telemetry = [], []
    for path in sorted(p for p in Path(root).iterdir() if (p/'run.json').exists()):
        run, profile, history, envs, fit_records = load_run(path)
        for env, requests in envs.items():
            label = {'run': path.name, 'env': env}
            for name, cfg in TWINS.items():
                for row, req in zip(overhead.time_twin(requests, cfg, profile, history, fit_records, repeats=repeats), requests):
                    times.append({**label, 'twin': name, **row, 'fu_seconds_full': planner.cost(req, planner.R_FULL),
                                  'fu_seconds_fixed': planner.cost(req, planner.R_FIXED)})
            for req in requests:
                clients = len(req['counts'])
                for option, rounds in (('full', planner.R_FULL), ('fixed', planner.R_FIXED)):
                    telemetry.append({**label, 'step': req['step'], 'option': option, 'rounds': rounds, 'clients': clients,
                                      **overhead.telemetry_bytes(*planner.observed_records(req, rounds)),
                                      'payload_bytes_raw': 2*rounds*clients*run['model_bytes']})
    return pd.DataFrame(times), pd.DataFrame(telemetry)


def summarize(times, telemetry):
    times = times.assign(dt_seconds=times.plan_seconds + times.sync_seconds)
    twins = {}
    for name, g in times.groupby('twin', sort=False):
        twins[name] = {'requests': int(len(g)),
                       'plan_microseconds': {'median': 1e6*g.plan_seconds.median(), 'p95': 1e6*g.plan_seconds.quantile(.95), 'max': 1e6*g.plan_seconds.max()},
                       'sync_microseconds': {'median': 1e6*g.sync_seconds.median(), 'p95': 1e6*g.sync_seconds.quantile(.95), 'max': 1e6*g.sync_seconds.max()},
                       'dt_seconds_total': float(g.dt_seconds.sum()), 'fu_seconds_total_full': float(g.fu_seconds_full.sum()),
                       'dt_share_of_fu_total': float(g.dt_seconds.sum()/g.fu_seconds_full.sum()),
                       # the worst single request: the twin's time against the cheapest operation it could have chosen
                       'dt_share_of_fu_worst_request': float((g.dt_seconds/g.fu_seconds_fixed).max())}
    tele = {}
    for option, g in telemetry.groupby('option', sort=False):
        tele[option] = {'rounds': int(g.rounds.iloc[0]), 'clients_per_shard': sorted(int(c) for c in g.clients.unique()),
                        **{k: {'median': float(g[k].median()), 'max': int(g[k].max())} for k in ('records_json', 'records_zlib', 'summary_json')},
                        'payload_bytes_raw_median': float(g.payload_bytes_raw.median()),
                        'share_of_payload': {k: float((g[k]/g.payload_bytes_raw).median()) for k in ('records_json', 'records_zlib', 'summary_json')}}
    return {'twins': twins, 'telemetry': tele}


def timed(fn, repeats):
    samples = []
    for _ in range(repeats):
        started = time.perf_counter(); fn(); samples.append(time.perf_counter() - started)
    return {'median_seconds': median(samples), 'max_seconds': max(samples), 'repeats': repeats}


def live(seed, shards, data_dir, threads, requests_measured, train_config):
    import torch
    from scripts.dt.machine import prefer_performance
    from scripts.dt.run_stress_trajectories import shard_counts, stream_requests
    from scripts.shard.train_config import CONFIGS
    from src.data.federated import partition
    from src.data.loading import load_data
    from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner, assign_shards, _py
    process = prefer_performance()
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)
    data = load_data(data_dir)
    parts = partition(data['train']['y'], 42, .5, 20)
    config = ShardConfig.from_train_config(CONFIGS[train_config], min_clients_design=4, merge_below=3)
    base = ShardedUnlearner(data['train'], data['validation'], parts, assign_shards(shards, 20), seed, 'shard_full', config)
    fit = base.fit()
    profile = planner.rates(fit['round_costs'], fit['client_costs'])
    state = {'shard_counts': {str(j): {'rows': int(sum(len(base.parts[c]) for c in a)),
                                       **timed(lambda j=j, a=a: shard_counts(base, j, a[0], 1), 50)}
                              for j, a in enumerate(base.assignment)},
             'client_state': {**timed(base.client_state, 10), 'clients': len(base.present),
                              'json_bytes': len(json.dumps(_py(base.client_state()), separators=(',', ':')))},
             'profile_json_bytes': len(json.dumps(profile, separators=(',', ':')))}
    # metering on / off: the same requests from the same state, order alternating
    forks = {'metered': base.fork('shard_full'), 'unmetered': base.fork('shard_full')}
    forks['unmetered'].measure_payload = False
    rows = []
    for k, r in enumerate(stream_requests(base, 'spread')[:requests_measured]):
        for name in (('metered', 'unmetered') if k % 2 == 0 else ('unmetered', 'metered')):
            fed = forks[name]
            started = time.perf_counter()
            result = fed.unlearn(ForgetRequest(r['client'], r['f']))
            wall = time.perf_counter() - started
            rows.append({'step': r['step'], 'shard': r['shard'], 'variant': name, 'operation_wall_seconds': wall,
                         'round_wall_seconds': sum(x['wall_seconds'] for x in result.round_costs),
                         'payload_pack_seconds': result.cost['payload_pack_seconds'], 'rounds': result.cost['rounds'],
                         'client_rounds': result.cost['client_rounds']})
    same = all(torch.equal(a, b) for j in forks['metered'].included()
               for a, b in zip(forks['metered'].shards[j].model.state_dict().values(),
                               forks['unmetered'].shards[j].model.state_dict().values()))
    frame = pd.DataFrame(rows)
    pivot = frame.pivot(index='step', columns='variant')
    outside = lambda v: pivot['operation_wall_seconds'][v] - pivot['round_wall_seconds'][v]
    metering = {'requests': int(pivot.shape[0]), 'weights_identical': bool(same),
                'round_wall_seconds_total': {v: float(pivot['round_wall_seconds'][v].sum()) for v in forks},
                'outside_rounds_seconds_total': {v: float(outside(v).sum()) for v in forks},
                'payload_pack_seconds_total': float(pivot['payload_pack_seconds']['metered'].sum()),
                'metering_seconds_per_client_round': float((outside('metered') - outside('unmetered')).sum()
                                                            / pivot['client_rounds']['metered'].sum()),
                'metering_share_of_round_time': float((outside('metered') - outside('unmetered')).sum()
                                                       / pivot['round_wall_seconds']['metered'].sum()),
                'per_request': rows}
    return _py({'seed': seed, 'shards': shards, 'train_config': train_config, 'process': process, 'state': state,
                'metering': metering})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--repeats', type=int, default=5)
    parser.add_argument('--live', action='store_true')
    parser.add_argument('--seed', type=int, default=50042)
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--requests', type=int, default=8, help='requests of the spread stream in the metering comparison')
    parser.add_argument('--data', type=Path, default=ROOT/'data/processed/connected_v3_auxfit_seed42')
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--train-config', default='standard', help=argparse.SUPPRESS)   # tests only
    args = parser.parse_args(argv)
    times, telemetry = offline(args.root, args.repeats)
    if times.empty:
        raise SystemExit(f'no trajectories in {args.root}')
    args.output.mkdir(parents=True, exist_ok=True)
    times.to_csv(args.output/'twin_times.csv', index=False, float_format='%.6g')
    telemetry.to_csv(args.output/'telemetry.csv', index=False)
    summary = summarize(times, telemetry)
    (args.output/'summary.json').write_text(json.dumps(summary, indent=1), encoding='utf-8')
    if args.live:
        (args.output/'live.json').write_text(json.dumps(live(args.seed, args.shards, args.data, args.threads, args.requests,
                                                             args.train_config), indent=1), encoding='utf-8')
    shown = {name: {'plan_us': round(v['plan_microseconds']['median'], 1), 'sync_us': round(v['sync_microseconds']['median'], 1),
                    'share_of_fu': v['dt_share_of_fu_total']} for name, v in summary['twins'].items()}
    print(json.dumps({'output': args.output.as_posix(), 'twins': shown, 'live': bool(args.live)}))


if __name__ == '__main__':
    main()
