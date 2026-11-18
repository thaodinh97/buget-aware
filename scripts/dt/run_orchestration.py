"""C1/C2 orchestration protocol (docs/dt_orchestration_plan.md): proposed controller vs literature baselines and
ablations, wall-clock arrivals, actual sharded exact unlearning.

    python scripts/dt/run_orchestration.py results/dt_orchestration --config configs/dt_orchestration.json

Per block (seed x family x load x environment): one fit, one request stream, one stress path; every arm forks the
same fitted ensemble and serves the same requests. After each unlearning operation the decision threshold is chosen
again on validation from the model just retrained (cfg['threshold_rule']); that time is part of the service time.
Utility is reported on calibration (threshold from validation); the test split is never loaded.
The lock hashes source files after CRLF -> LF, so a Windows checkout with core.autocrlf matches a Linux one.
--resume accepts only the same config/source and skips completed blocks; partial blocks are never overwritten.
"""
import argparse
import hashlib
import json
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from scripts.dt.machine import prefer_performance
from scripts.dt.run_batch_dt import weights_sha
from scripts.dt.run_online_confirmation import condition_at, stream_config, write_json
from scripts.dt.run_predictive_confirmation import keep_awake
from scripts.dt.run_stress_trajectories import stream_requests
from scripts.shard.fpr_threshold_check import choose as choose_threshold, sweep
from scripts.shard.shard_v2 import data_content_sha256
from scripts.shard.shard_v3 import EXPECTED_CLIENT_ROWS, load_protocol_data
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.data.prepare_log1p import check_numeric_transform
from src.dt import orchestration, predictive, realtime
from src.dt.stress import install
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner, _py, assign_shards

# name -> (kind, options). 'queue' = proposed controller and C2 ablations, 'baseline' = C1 literature baselines,
# 'legacy' = the earlier controller of this repo (src/dt/realtime.py with a row-rate EWMA twin).
ARMS = {
    'dt_queue': ('queue', dict(hybrid=True)),
    'rows_queue': ('queue', dict(hybrid=False)),
    'queue_edf_order': ('queue', dict(hybrid=True, order='edf')),
    'queue_no_overhead': ('queue', dict(hybrid=True, overhead=False)),
    'queue_no_pooling': ('queue', dict(hybrid=True, pooling=False)),
    'immediate': ('baseline', dict(mode='immediate')),
    'window': ('baseline', dict(mode='window')),
    'lazy': ('baseline', dict(mode='lazy')),
    'edf': ('baseline', dict(mode='edf')),
    'rows_legacy': ('legacy', {}),
}
RESERVED_SEEDS = set(range(60042, 60052)) | {71042, 71043, 71044, 72042, 72043, 72044}
METRICS = ('macro_f1', 'fpr', 'recall_attack')


def lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def source_hashes(config_path, plan_path):
    paths = list((ROOT/'src').rglob('*.py')) + list((ROOT/'scripts/dt').glob('*.py'))
    paths += [ROOT/'scripts/shard'/p for p in ('shard_v2.py', 'shard_v3.py', 'stream_v3.py', 'train_config.py',
                                                'shard_dt.py', 'fpr_threshold_check.py')]
    paths += [Path(config_path).resolve(), Path(plan_path).resolve()]
    return {p.resolve().relative_to(ROOT).as_posix(): lf_sha256(p) for p in sorted(set(paths))}


def threshold(fed, rule):
    margin = fed.margin(fed.validation['X'])
    return choose_threshold(rule, sweep(margin, fed.validation['y']))


def utility(fed, split, t):
    margin = fed.margin(split['X'])
    tuned, default = detector_metrics(split['y'], margin, t), detector_metrics(split['y'], margin, 0.)
    return {**{k: tuned[k] for k in METRICS}, **{f'{k}_default_threshold': default[k] for k in METRICS}}


def run_arm(base, fit, requests, name, env, cfg, repeat, budget, deadline, window, calibration, directory):
    kind, options = ARMS[name]
    directory.mkdir(parents=True, exist_ok=False)
    fed = base.fork('shard_full')
    controller = install(fed, lambda t: condition_at(env, t, requests[-1]['time'], fed.assignment_initial, cfg, repeat))

    def counts(shard, batch):
        result = {int(c): int(fed.alive[fed.parts[c]].sum()) for c in fed.assignment[shard]}
        for r in batch:
            result[r['client']] -= r['rows']
        return result

    if kind == 'legacy':
        twin = realtime.OnlineTwin('rows_ewma', fit['round_costs'], fit['client_costs'], counts,
                                   cfg['alpha'], cfg['margin_q'], cfg['margin_window'])
    else:
        twin = predictive.ServiceTwin(fit, counts, hybrid=options.get('hybrid', True), q=cfg['margin_q'],
                                      window=cfg['margin_window'], loss_alpha=cfg['expert_loss_alpha'],
                                      eta=cfg['expert_eta'], floor=cfg['expert_floor'])
    state = {'threshold': threshold(fed, cfg['threshold_rule']), 'threshold_seconds': 0.}

    def execute(batch, policy, started):
        controller.step = started
        cond = controller.condition()
        result = fed.unlearn_batch([ForgetRequest(r['client'], r['f']) for r in batch], policy)
        tick = time.perf_counter()
        state['threshold'] = threshold(fed, cfg['threshold_rule'])   # from the model just retrained
        spent = time.perf_counter()-tick
        state['threshold_seconds'] += spent
        # Returning is the model-ready boundary: retrained shards and their threshold are in place.
        return dict(round_seconds=sum(r['wall_seconds'] for r in result.round_costs),
                    round_costs=result.round_costs, client_costs=result.client_costs,
                    rows_x_rounds=result.cost['rows_x_rounds'], payload_bytes=result.cost['train_bytes_raw'],
                    exact=result.exact, structure_ok=result.structure_ok, slow_factors=cond.slow,
                    bandwidth_bps=cond.bandwidth_bps if np.isfinite(cond.bandwidth_bps) else None,
                    threshold=state['threshold'], threshold_seconds=spent)

    def log(event):
        with (directory/'events.jsonl').open('a', encoding='utf-8') as handle:
            handle.write(json.dumps(_py(event), allow_nan=False)+'\n')
        print(json.dumps(dict(arm=name, event=event['event'], served=event['steps'], policy=event['policy'],
                              ready=round(event['model_ready_seconds'], 2), misses=event['deadline_misses'])), flush=True)

    full, fixed = base.r_base, base.config.shard_fixed_rounds
    if kind == 'queue':
        events = orchestration.queue_run(requests, twin, execute, budget, deadline, full, fixed,
                                         order=options.get('order', 'enumerate'),
                                         overhead=options.get('overhead', True),
                                         pooling=options.get('pooling', True), on_event=log)
    elif kind == 'baseline':
        events = orchestration.baseline_run(requests, twin, execute, budget, deadline, options['mode'],
                                            window=window, full_rounds=full, fixed_rounds=fixed, on_event=log)
    else:
        events = realtime.run(requests, twin, execute, budget, deadline, full, fixed,
                              cfg['service_padding_seconds'], on_event=log)
    del fed.worker_factory
    # Audits run after all measured requests; they never affect latency or forecasts.
    check = fed.verify_bit_exact()
    row = realtime.summarize(events)
    row.update(predictor=name, kind=kind, exact_all_events=all(e['exact'] and e['structure_ok'] for e in events),
               bit_exact=bool(check['bit_exact'] and check['structure_ok'] and check['included_equals_active']),
               alive_rows=int(fed.alive.sum()), threshold_final=state['threshold'],
               threshold_seconds_total=state['threshold_seconds'],
               plan_seconds_total=sum(e['plan_seconds'] for e in events),
               sync_seconds_total=sum(e['sync_seconds'] for e in events),
               **{f'validation_{k}': v for k, v in utility(fed, fed.validation, state['threshold']).items()},
               **{f'calibration_{k}': v for k, v in utility(fed, calibration, state['threshold']).items()})
    write_json(directory/'final.json', dict(summary=row, verify_bit_exact=check, weights_sha256=weights_sha(fed),
               shard_rounds={str(j): s.rounds for j, s in enumerate(fed.shards) if s}))
    if not (row['bit_exact'] and row['exact_all_events']):
        raise RuntimeError(f'exactness failed: {directory}')
    return row


def blocks(cfg):
    """(repeat, seed, family, load, environment) in the counterbalanced run order."""
    out = []
    for repeat, seed in enumerate(cfg['seeds']):
        for family_i, family in enumerate(cfg['families']):
            for load_i, load in enumerate(cfg['loads']):
                offset = (repeat+family_i+load_i) % len(cfg['environments'])
                for env in cfg['environments'][offset:]+cfg['environments'][:offset]:
                    out.append((repeat, seed, family, load, env))
    return out


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    parser.add_argument('--config', type=Path, default=ROOT/'configs/dt_orchestration.json')
    parser.add_argument('--plan', type=Path, default=ROOT/'docs/dt_orchestration_plan.md')
    parser.add_argument('--resume', action='store_true')
    args = parser.parse_args(argv)
    cfg = json.loads(args.config.read_text(encoding='utf-8'))
    if set(cfg['seeds']) & RESERVED_SEEDS:
        raise ValueError('seeds of the v3 and earlier DT confirmations are reserved')
    if set(cfg['arms']) - set(ARMS) or len(set(cfg['arms'])) != len(cfg['arms']):
        raise ValueError(f'unknown or repeated arms: {cfg["arms"]}')
    hashes = source_hashes(args.config, args.plan)
    out = args.output
    if out.exists():
        if not args.resume:
            raise FileExistsError(out)
        lock = json.loads((out/'protocol_lock.json').read_text(encoding='utf-8'))
        if lock['config'] != cfg or lock['source_sha256_lf'] != hashes:
            changed = sorted(k for k in set(hashes) | set(lock['source_sha256_lf'])
                             if hashes.get(k) != lock['source_sha256_lf'].get(k))
            raise ValueError(f'config/source differs from the frozen run: {changed}')
    data_dir = ROOT/cfg['data_dir']
    check_numeric_transform(data_dir, cfg['numeric_transform'])
    data, manifest = load_protocol_data(data_dir)
    content = data_content_sha256(data)
    if not out.exists():
        out.mkdir(parents=True)
        write_json(out/'protocol_lock.json', dict(config=cfg, source_sha256_lf=hashes, data_content_sha256=content,
            data_artifacts_sha256=manifest['artifacts_sha256'],
            frozen_utc=time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime()),
            machine=dict(platform=platform.platform(), python=platform.python_version(), torch=torch.__version__,
                         numpy=np.__version__), system_sleep_prevention=sys.platform == 'win32'))
    lock = json.loads((out/'protocol_lock.json').read_text(encoding='utf-8'))
    if lock['data_content_sha256'] != content:
        raise ValueError('data content differs from the frozen run')
    process = prefer_performance()
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    parts = partition(data['train']['y'], 42, .5, 20)
    if [len(p) for p in parts] != EXPECTED_CLIENT_ROWS:
        raise ValueError('client row counts differ from the frozen partition')
    config = ShardConfig.from_train_config(CONFIGS[cfg['train_config']], min_clients_design=4, merge_below=3)
    summaries = []
    for index, (repeat, seed, family, load, env) in enumerate(blocks(cfg)):
        block_name = f'r{repeat}_{family}_load{load:g}_{env}'
        block = out/block_name
        if (block/'complete.json').exists():
            summaries.extend(json.loads((block/'complete.json').read_text(encoding='utf-8'))['summaries'])
            continue
        if block.exists():
            raise RuntimeError(f'partial block preserved; archive it before resuming: {block}')
        block.mkdir()
        print(json.dumps(dict(block=block_name, status='fit')), flush=True)
        base = ShardedUnlearner(data['train'], data['validation'], parts, assign_shards(cfg['shards'], 20),
                               seed, 'shard_full', config)
        fit = base.fit()
        pd.DataFrame(fit['round_costs']).to_csv(block/'fit_round_costs.csv', index=False)
        if max(r['wall_seconds'] for r in fit['round_costs']) > cfg['calibration_max_round_seconds']:
            write_json(block/'invalid_calibration.json', dict(reason='individual fit round > predeclared limit'))
            raise RuntimeError('calibration interruption/slowdown: inspect the saved fit before any online run')
        reference = max(sum(r['wall_seconds'] for r in fit['round_costs'] if r['shard'] == j)
                        for j in range(cfg['shards']))
        budget, deadline = cfg['budget_scale']*reference, cfg['deadline_scale']*reference
        window = cfg['window_scale']*reference
        scfg = stream_config(cfg, repeat)
        requests = stream_requests(base, family, scfg)
        for r in requests:
            r['time'] *= load*reference
        offset = index % len(cfg['arms'])
        arms = cfg['arms'][offset:]+cfg['arms'][:offset]
        write_json(block/'run.json', dict(repeat=repeat, seed=seed, family=family, load=load, environment=env,
            reference_full_seconds=reference, budget_seconds=budget, deadline_seconds=deadline, window_seconds=window,
            requests=[{**r, 'f': r['f'].tolist()} for r in requests], stream_config=scfg, arm_order=arms,
            process=process, data_content_sha256=content, initial_weights_sha256=weights_sha(base)))
        rows = []
        for position, name in enumerate(arms):
            print(json.dumps(dict(block=block_name, arm=name, status='start')), flush=True)
            started = time.perf_counter()
            row = run_arm(base, fit, requests, name, env, cfg, repeat, budget, deadline, window,
                          data['calibration'], block/name)
            row.update(block=block_name, repeat=repeat, seed=seed, family=family, load=load, environment=env,
                       position=position, budget_seconds=budget, deadline_seconds=deadline,
                       arm_wall_seconds=time.perf_counter()-started)
            rows.append(row)
        if len({r['alive_rows'] for r in rows}) != 1:
            raise RuntimeError('arms removed different amounts of data')
        write_json(block/'complete.json', dict(summaries=rows))
        summaries.extend(rows)
        pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
        print(json.dumps(dict(block=block_name, status='complete')), flush=True)
    pd.DataFrame(summaries).to_csv(out/'summary.csv', index=False)
    write_json(out/'complete.json', dict(blocks=len(summaries)//len(cfg['arms']), arms=len(summaries),
                                       source_hashes_unchanged=source_hashes(args.config, args.plan) == hashes))


if __name__ == '__main__':
    with keep_awake():
        main()
