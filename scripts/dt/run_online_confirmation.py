"""Prospectively configured online comparison; wall-clock arrivals, actual FU.

python scripts/dt/run_online_confirmation.py results/dt_online_confirmation
The config and source hashes are locked before any measured fit. --resume only
accepts the same files and skips completed arms, never overwrites partial arms.
All three predictors use identical scheduling rules. No counterfactual runs or
hidden escalation supply feedback. Every request is eventually served and costed.
"""
import argparse
import hashlib
import json
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
from scripts.dt.run_batch_dt import weights_sha
from scripts.dt.run_stress_trajectories import stream_requests
from scripts.shard.shard_v2 import DATA_DEFAULT, DATA_SHA256, data_content_sha256
from scripts.shard.stream_v3 import STREAM
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.dt import realtime
from src.dt.stress import Condition, install, slow_clients
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner, _py, assign_shards


def write_json(path, value):
    Path(path).write_text(json.dumps(_py(value), indent=2, allow_nan=False), encoding='utf-8')


def source_hashes(config_path):
    paths = list((ROOT/'src').rglob('*.py')) + list((ROOT/'scripts/dt').glob('*.py'))
    paths += [ROOT/'scripts/shard'/p for p in ('shard_v2.py', 'stream_v3.py', 'train_config.py', 'shard_dt.py')]
    paths += [Path(config_path)]
    return {p.relative_to(ROOT).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(set(paths))}


def stream_config(cfg, repeat):
    result = dict(STREAM, K=cfg['requests'])
    for offset, field in enumerate(('seed_size', 'seed_time', 'seed_choice', 'seed_rank', 'seed_rows')):
        result[field] = cfg['stream_seed_base'] + repeat * 100 + offset
    return result


def condition_at(env, seconds, horizon, assignment, cfg, repeat):
    """External stress path, shared across arms, sampled at operation start.

    During an operation its sampled condition is held constant. This is injected
    CPU/transfer delay on one machine, not a live distributed network.
    """
    if env == 'none':
        return Condition()
    if env == 'gradual_clients':
        progress = float(np.clip((seconds / max(horizon, .001) - .2) / .6, 0., 1.))
        factor = 1 + (cfg['slow_factor_max'] - 1) * progress
        return Condition(slow={c: factor for c in slow_clients(assignment, cfg['stress_seed'] + repeat)})
    if env != 'jitter_bandwidth':
        raise ValueError(env)
    if seconds < .2 * horizon:
        return Condition()
    bucket = int(seconds / cfg['noise_bucket_seconds'])
    rng = np.random.default_rng([cfg['stress_seed'], repeat, bucket])
    bandwidth = rng.uniform(*cfg['bandwidth_mbps_range']) * 1e6
    return Condition(bandwidth_bps=float(bandwidth))


def run_arm(base, fit, requests, name, env, cfg, repeat, budget, deadline, directory):
    directory.mkdir(parents=True, exist_ok=False)
    fed = base.fork('shard_full')
    controller = install(fed, lambda t: condition_at(env, t, requests[-1]['time'],
                                                    fed.assignment_initial, cfg, repeat))

    def counts(shard, batch):
        result = {int(c): int(fed.alive[fed.parts[c]].sum()) for c in fed.assignment[shard]}
        for r in batch:
            result[r['client']] -= r['rows']
        return result

    twin = realtime.OnlineTwin(name, fit['round_costs'], fit['client_costs'], counts,
                               cfg['alpha'], cfg['margin_q'], cfg['margin_window'])

    def execute(batch, policy, started):
        controller.step = started
        condition = controller.condition()
        result = fed.unlearn_batch([ForgetRequest(r['client'], r['f']) for r in batch], policy)
        # Returning from unlearn_batch is the in-process model-ready boundary.
        return dict(round_seconds=sum(r['wall_seconds'] for r in result.round_costs),
                    round_costs=result.round_costs, client_costs=result.client_costs,
                    rows_x_rounds=result.cost['rows_x_rounds'], payload_bytes=result.cost['train_bytes_raw'],
                    exact=result.exact, structure_ok=result.structure_ok,
                    slow_factors=condition.slow,
                    bandwidth_bps=condition.bandwidth_bps if np.isfinite(condition.bandwidth_bps) else None)

    def log(event):
        with (directory/'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(_py(event), allow_nan=False) + '\n')
        print(json.dumps(dict(arm=name, event=event['event'], served=event['steps'],
                             ready=round(event['model_ready_seconds'], 2), misses=event['deadline_misses'])), flush=True)

    events = realtime.run(requests, twin, execute, budget, deadline, base.r_base,
                          base.config.shard_fixed_rounds, cfg['service_padding_seconds'], on_event=log)
    del fed.worker_factory
    # Audits occur after ALL measured requests, never affect queue latency or forecasts.
    check = fed.verify_bit_exact()
    metrics = detector_metrics(fed.validation['y'], fed.predict_proba(fed.validation['X']))
    summary = realtime.summarize(events)
    summary.update(predictor=name, exact_all_events=all(e['exact'] and e['structure_ok'] for e in events),
                   bit_exact=bool(check['bit_exact'] and check['structure_ok'] and check['included_equals_active']),
                   validation_macro_f1=metrics['macro_f1'], validation_fpr=metrics['fpr'],
                   alive_rows=int(fed.alive.sum()))
    write_json(directory/'final.json', dict(summary=summary, verify_bit_exact=check,
               weights_sha256=weights_sha(fed), shard_rounds={str(j): s.rounds for j, s in enumerate(fed.shards) if s}))
    if not (summary['bit_exact'] and summary['exact_all_events']):
        raise RuntimeError(f'exactness failed: {directory}')
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/dt_online_confirmation.json')
    parser.add_argument('--data', type=Path, default=ROOT/DATA_DEFAULT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if any(60042 <= seed <= 60051 for seed in cfg['seeds']):
        raise ValueError('reserved v3 seeds must not be used')
    hashes = source_hashes(args.config)
    out = args.output
    if out.exists():
        if not args.resume:
            raise FileExistsError(out)
        locked = json.loads((out/'protocol_lock.json').read_text())
        if locked['config'] != cfg or locked['source_sha256'] != hashes:
            raise ValueError('config/source differs from frozen run')
    else:
        out.mkdir(parents=True)
        write_json(out/'protocol_lock.json', dict(config=cfg, source_sha256=hashes,
                   frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
                   machine=dict(platform=platform.platform(), python=platform.python_version(),
                                torch=torch.__version__, numpy=np.__version__)))
    process = prefer_performance()
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    from src.data.loading import load_data
    data = load_data(args.data)
    content_hash = data_content_sha256(data)
    if content_hash != DATA_SHA256:
        raise ValueError('dataset hash differs from the fixed shard dataset')
    parts = partition(data['train']['y'], 42, .5, 20)
    train_cfg = ShardConfig.from_train_config(CONFIGS[cfg['train_config']], min_clients_design=4, merge_below=3)
    summaries = []
    for repeat, seed in enumerate(cfg['seeds']):
        for family_i, family in enumerate(cfg['families']):
            # Counterbalance condition order as well as arm order.
            offset = (repeat + family_i) % len(cfg['environments'])
            envs = cfg['environments'][offset:] + cfg['environments'][:offset]
            for env in envs:
                env_i = cfg['environments'].index(env)
                block_name = f'r{repeat}_{family}_{env}'
                block = out/block_name
                if (block/'complete.json').exists():
                    summaries.extend(json.loads((block/'complete.json').read_text())['summaries'])
                    continue
                if block.exists():
                    raise RuntimeError(f'partial block preserved; use a new output for a fresh run: {block}')
                block.mkdir()
                print(json.dumps(dict(block=block_name, status='fit')), flush=True)
                base = ShardedUnlearner(data['train'], data['validation'], parts,
                                       assign_shards(cfg['shards'], 20), seed, 'shard_full', train_cfg)
                fit = base.fit()
                reference = max(sum(r['wall_seconds'] for r in fit['round_costs'] if r['shard'] == j)
                                for j in range(cfg['shards']))
                budget = cfg['budget_scale'] * reference
                deadline = cfg['deadline_scale'] * reference
                scfg = stream_config(cfg, repeat)
                requests = stream_requests(base, family, scfg)
                for r in requests:
                    r['time'] *= cfg['mean_interarrival_scale'] * reference
                arm_offset = (repeat + family_i + env_i) % len(cfg['predictors'])
                arms = cfg['predictors'][arm_offset:] + cfg['predictors'][:arm_offset]
                write_json(block/'run.json', dict(repeat=repeat, seed=seed, family=family, environment=env,
                    reference_full_seconds=reference, budget_seconds=budget, deadline_seconds=deadline,
                    stream_config=scfg, requests=[{**r, 'f': r['f'].tolist()} for r in requests],
                    arm_order=arms, process=process, data_content_sha256=content_hash,
                    initial_weights_sha256=weights_sha(base)))
                pd.DataFrame(fit['round_costs']).to_csv(block/'fit_round_costs.csv', index=False)
                pd.DataFrame(fit['client_costs']).to_csv(block/'fit_client_costs.csv', index=False)
                block_rows = []
                for position, name in enumerate(arms):
                    print(json.dumps(dict(block=block_name, arm=name, status='start')), flush=True)
                    row = run_arm(base, fit, requests, name, env, cfg, repeat, budget, deadline, block/name)
                    row.update(block=block_name, repeat=repeat, seed=seed, family=family, environment=env,
                               position=position, budget_seconds=budget, deadline_seconds=deadline)
                    block_rows.append(row)
                if len({r['alive_rows'] for r in block_rows}) != 1:
                    raise RuntimeError('arms did not remove the same number of rows')
                write_json(block/'complete.json', dict(summaries=block_rows))
                summaries.extend(block_rows)
                pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
                print(json.dumps(dict(block=block_name, status='complete')), flush=True)
    pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
    write_json(out/'complete.json', dict(blocks=len(summaries)//len(cfg['predictors']), arms=len(summaries),
                                       source_hashes_unchanged=source_hashes(args.config) == hashes))


if __name__ == '__main__':
    main()
