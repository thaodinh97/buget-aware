"""New confirmation for the experimental service/queue-aware twin.

Runs the new twin, a rows-only ablation with the SAME new controller, and the
strongest legacy baseline. All settings frozen before any fit; old results/code
remain unchanged. Windows sleep prevention is scoped to this process.
"""
import argparse
import contextlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd
import torch

from scripts.dt.machine import prefer_performance
from scripts.dt.run_batch_dt import weights_sha
from scripts.dt.run_online_confirmation import (condition_at, source_hashes, stream_config, write_json,
                                                 run_arm as run_legacy_arm)
from scripts.dt.run_stress_trajectories import stream_requests
from scripts.shard.shard_v2 import DATA_DEFAULT, DATA_SHA256, data_content_sha256
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.dt import predictive, realtime
from src.dt.stress import install
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner, _py, assign_shards


@contextlib.contextmanager
def keep_awake():
    if sys.platform != 'win32':
        yield
        return
    import ctypes
    kernel = ctypes.WinDLL('kernel32', use_last_error=True)
    kernel.SetThreadExecutionState.argtypes = [ctypes.c_uint]
    kernel.SetThreadExecutionState.restype = ctypes.c_uint
    if not kernel.SetThreadExecutionState(0x80000001):
        raise ctypes.WinError(ctypes.get_last_error())
    try:
        yield
    finally:
        kernel.SetThreadExecutionState(0x80000000)


def run_arm(base, fit, requests, name, env, cfg, repeat, budget, deadline, directory):
    if name == 'rows_legacy':
        row = run_legacy_arm(base, fit, requests, 'rows_ewma', env, cfg, repeat, budget, deadline, directory)
        row['predictor'] = name
        final = json.loads((directory/'final.json').read_text())
        final['summary']['predictor'] = name
        write_json(directory/'final.json', final)
        return row
    if name not in ('dt_queue', 'rows_queue'):
        raise ValueError(name)
    directory.mkdir(parents=True, exist_ok=False)
    fed = base.fork('shard_full')
    controller = install(fed, lambda t: condition_at(env, t, requests[-1]['time'],
                                                    fed.assignment_initial, cfg, repeat))

    def counts(shard, batch):
        result = {int(c): int(fed.alive[fed.parts[c]].sum()) for c in fed.assignment[shard]}
        for r in batch:
            result[r['client']] -= r['rows']
        return result

    twin = predictive.ServiceTwin(fit, counts, hybrid=name == 'dt_queue', q=cfg['margin_q'],
                                   window=cfg['margin_window'], loss_alpha=cfg['expert_loss_alpha'],
                                   eta=cfg['expert_eta'], floor=cfg['expert_floor'])

    def execute(batch, policy, started):
        controller.step = started
        cond = controller.condition()
        result = fed.unlearn_batch([ForgetRequest(r['client'], r['f']) for r in batch], policy)
        return dict(round_seconds=sum(r['wall_seconds'] for r in result.round_costs),
                    round_costs=result.round_costs, client_costs=result.client_costs,
                    rows_x_rounds=result.cost['rows_x_rounds'], payload_bytes=result.cost['train_bytes_raw'],
                    exact=result.exact, structure_ok=result.structure_ok, slow_factors=cond.slow,
                    bandwidth_bps=cond.bandwidth_bps if cond.bandwidth_bps != float('inf') else None)

    def log(event):
        with (directory/'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(_py(event), allow_nan=False)+'\n')
        print(json.dumps(dict(arm=name, event=event['event'], served=event['steps'],
                             ready=round(event['model_ready_seconds'], 2), misses=event['deadline_misses'])), flush=True)

    events = predictive.run(requests, twin, execute, budget, deadline, base.r_base,
                            base.config.shard_fixed_rounds, on_event=log)
    del fed.worker_factory
    check = fed.verify_bit_exact()
    metrics = detector_metrics(fed.validation['y'], fed.predict_proba(fed.validation['X']))
    row = realtime.summarize(events)
    row.update(predictor=name, exact_all_events=all(e['exact'] and e['structure_ok'] for e in events),
               bit_exact=bool(check['bit_exact'] and check['structure_ok'] and check['included_equals_active']),
               validation_macro_f1=metrics['macro_f1'], validation_fpr=metrics['fpr'], alive_rows=int(fed.alive.sum()),
               plan_seconds_total=sum(e['plan_seconds'] for e in events),
               sync_seconds_total=sum(e['sync_seconds'] for e in events))
    write_json(directory/'final.json', dict(summary=row, verify_bit_exact=check, weights_sha256=weights_sha(fed),
        shard_rounds={str(j): s.rounds for j, s in enumerate(fed.shards) if s}, expert_names=twin.names,
        final_expert_weights=twin.weights().tolist()))
    if not (row['bit_exact'] and row['exact_all_events']):
        raise RuntimeError(f'exactness failed: {directory}')
    return row


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/dt_predictive_confirmation.json')
    parser.add_argument('--data', type=Path, default=ROOT/DATA_DEFAULT)
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if any(60042 <= seed <= 60051 or 71042 <= seed <= 71044 for seed in cfg['seeds']):
        raise ValueError('reserved v3 and previously observed confirmation seeds are excluded')
    hashes = source_hashes(args.config.resolve())
    out = args.output
    if out.exists():
        if not args.resume:
            raise FileExistsError(out)
        lock = json.loads((out/'protocol_lock.json').read_text())
        if lock['config'] != cfg or lock['source_sha256'] != hashes:
            raise ValueError('code/config differs from frozen run')
    else:
        out.mkdir(parents=True)
        write_json(out/'protocol_lock.json', dict(config=cfg, source_sha256=hashes,
            frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            machine=dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__),
            development_sources=['results/dt_online_confirmation', 'results/dt_extensions'],
            system_sleep_prevention=sys.platform == 'win32'))
    process = prefer_performance()
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    from src.data.loading import load_data
    data = load_data(args.data)
    content = data_content_sha256(data)
    if content != DATA_SHA256:
        raise ValueError('dataset hash mismatch')
    parts = partition(data['train']['y'], 42, .5, 20)
    config = ShardConfig.from_train_config(CONFIGS[cfg['train_config']], min_clients_design=4, merge_below=3)
    summaries = []
    for repeat, seed in enumerate(cfg['seeds']):
        for family_i, family in enumerate(cfg['families']):
            offset = (repeat+family_i) % len(cfg['environments'])
            envs = cfg['environments'][offset:]+cfg['environments'][:offset]
            for env in envs:
                env_i = cfg['environments'].index(env)
                block_name = f'r{repeat}_{family}_{env}'
                block = out/block_name
                if (block/'complete.json').exists():
                    summaries.extend(json.loads((block/'complete.json').read_text())['summaries'])
                    continue
                if block.exists():
                    raise RuntimeError(f'partial block preserved; archive it before resuming: {block}')
                block.mkdir()
                print(json.dumps(dict(block=block_name, status='fit')), flush=True)
                base = ShardedUnlearner(data['train'], data['validation'], parts, assign_shards(cfg['shards'], 20),
                                       seed, 'shard_full', config)
                fit = base.fit()
                pd.DataFrame(fit['round_costs']).to_csv(block/'fit_round_costs.csv', index=False)
                pd.DataFrame(fit['client_costs']).to_csv(block/'fit_client_costs.csv', index=False)
                if max(r['wall_seconds'] for r in fit['round_costs']) > cfg['calibration_max_round_seconds']:
                    write_json(block/'invalid_calibration.json', dict(reason='individual fit round > predeclared limit'))
                    raise RuntimeError('calibration interruption/slowdown: inspect saved fit before any online run')
                reference = max(sum(r['wall_seconds'] for r in fit['round_costs'] if r['shard'] == j)
                                for j in range(cfg['shards']))
                budget, deadline = cfg['budget_scale']*reference, cfg['deadline_scale']*reference
                scfg = stream_config(cfg, repeat)
                requests = stream_requests(base, family, scfg)
                for r in requests:
                    r['time'] *= cfg['mean_interarrival_scale']*reference
                offset = (repeat+family_i+env_i) % len(cfg['predictors'])
                arms = cfg['predictors'][offset:]+cfg['predictors'][:offset]
                write_json(block/'run.json', dict(repeat=repeat, seed=seed, family=family, environment=env,
                    reference_full_seconds=reference, budget_seconds=budget, deadline_seconds=deadline,
                    requests=[{**r, 'f': r['f'].tolist()} for r in requests], stream_config=scfg, arm_order=arms,
                    process=process, data_content_sha256=content, initial_weights_sha256=weights_sha(base)))
                block_rows = []
                for position, name in enumerate(arms):
                    print(json.dumps(dict(block=block_name, arm=name, status='start')), flush=True)
                    row = run_arm(base, fit, requests, name, env, cfg, repeat, budget, deadline, block/name)
                    row.update(block=block_name, repeat=repeat, seed=seed, family=family, environment=env,
                               position=position, budget_seconds=budget, deadline_seconds=deadline)
                    block_rows.append(row)
                if len({r['alive_rows'] for r in block_rows}) != 1:
                    raise RuntimeError('arms removed different data amounts')
                write_json(block/'complete.json', dict(summaries=block_rows))
                summaries.extend(block_rows)
                pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
                print(json.dumps(dict(block=block_name, status='complete')), flush=True)
    pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
    write_json(out/'complete.json', dict(blocks=len(summaries)//len(cfg['predictors']), arms=len(summaries),
                                       source_hashes_unchanged=source_hashes(args.config.resolve()) == hashes))


if __name__ == '__main__':
    with keep_awake():
        main()
