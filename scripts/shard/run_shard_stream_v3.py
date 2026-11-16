"""Run one family of the v3 sample-level shard protocol.

    SHARD_DESIGN=v3 SHARD_V3_FAMILY=spread FL_DEVICE=cpu python scripts/shard/run_shard_stream_v3.py ROOT/stream_spread

SHARD_DESIGN and SHARD_V3_FAMILY have no default, and the output directory must be
named stream_<family>; otherwise the CLI stops before creating anything.
The CLI runs a fresh process, and retries the entire family once if the initial
ensemble cannot be reproduced. Both attempt directories and records are kept.
run_experiment() is importable for a synthetic, explicitly non-confirmatory smoke.
"""
import argparse
import copy
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.shard.shard_v2 import assign, data_content_sha256
from scripts.shard.shard_v3 import (CONFIGS, DATA_SHA256, DESIGNS, EXPECTED_CLIENT_ROWS, RUNNER, SMOKE_CONFIG,
                                   machine_info, run_selection, source_hashes, start_run, write_json)
from scripts.shard.stream_v3 import (FAMILIES, STREAM, batch_windows, plan_stream, select_rows)
from src.data.federated import ids_hash, partition
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner, warm_up

POLICIES = ['none', 'shard_full', 'shard_fixed', 'utility_stop', 'drop_backfill', 'batch_full_2', 'batch_full_5']
METRICS = ('cal_macro_f1', 'cal_fpr', 'cal_recall_attack', 'client_f1_min', 'client_f1_mean')


def weights_equal(a, b):
    """Compare all serving shards, their row sets, rounds, seeds and tensors."""
    if a.included() != b.included() or a.assignment != b.assignment or not np.array_equal(a.alive, b.alive):
        return False
    for j in a.included():
        x, y = a.shards[j], b.shards[j]
        if x.seed != y.seed or x.rounds != y.rounds or not np.array_equal(x.idx, y.idx):
            return False
        xs, ys = x.model.state_dict(), y.model.state_dict()
        if xs.keys() != ys.keys() or not all(torch.equal(xs[k], ys[k]) for k in xs):
            return False
    return True


def report(fed, data, parts, alive):
    if not fed.included():
        return {'unavailable': True, **{key: float('nan') for key in METRICS}}
    ca, tr = data['calibration'], data['train']
    cal = detector_metrics(ca['y'], fed.predict_proba(ca['X']))
    pred = fed.predict_proba(tr['X'])
    f1 = [detector_metrics(tr['y'][p[alive[p]]], pred[p[alive[p]]])['macro_f1']
          for p in parts if alive[p].any()]
    return {'unavailable': False, **{f'cal_{k}': cal[k] for k in ('macro_f1', 'fpr', 'recall_attack')},
            'client_f1_min': float(np.nanmin(f1)), 'client_f1_mean': float(np.nanmean(f1))}


def materialize_stream(family, parts, assignment, train, stream_config):
    plan = plan_stream(family, [len(p) for p in parts], assignment, stream_config)
    alive = np.ones(len(train['y']), bool)
    requests = []
    for request in plan:
        p = parts[request['client']]
        selected = select_rows(p[alive[p]], request['rows'], request['step'], stream_config['seed_rows'])
        alive[selected] = False
        requests.append({**request, 'f': selected, 'forget_request_hash': ids_hash(train['row_id'][selected])})
    return requests


def _zero_cost():
    return {'rows_x_rounds': 0, 'rounds': 0, 'bytes': 0, 'seconds': 0., 'bytes_packed': 0}


def _charge(cumulative, result):
    cost = result.cost if hasattr(result, 'cost') else result['cost']
    delta = {'rows_x_rounds': cost['rows_x_rounds'], 'rounds': cost['rounds'],
             'bytes': cost['train_bytes_raw'], 'seconds': cost['wall_seconds_total'],
             'bytes_packed': (cost['bytes_down_packed'] or 0) + (cost['bytes_up_packed'] or 0)}
    for key in cumulative:
        cumulative[key] += delta[key]
    return {f'delta_{key}': value for key, value in delta.items()}


def _row(fed, data, parts, alive, context, request, kind, cumulative, delta=None, result=None, **extra):
    exact = fed.verify_exact() if context['policy'] != 'none' else {'exact': False, 'structure_ok': False}
    row = {**context, 'step': request['step'], 'kind': kind, 'time': request['time'],
           'client': request.get('client', -1), 'shard': request.get('shard', -1),
           'requested': request.get('requested', 0), 'rows': request.get('rows', 0),
           'forget_request_hash': request.get('forget_request_hash', ''),
           'processed_at': request['time'], 'latency': 0., 'exposure': 0., 'pending_rows': 0,
           **exact, 'exactness_scope': 'conditional_on_round_count' if context['policy'] == 'utility_stop' else 'unconditional',
           'shards_included': len(fed.included()), 'included_shards': json.dumps(fed.included()),
           'pending': len(fed.pending), 'fallback_rounds': 0, 'condition_met': np.nan,
           **cumulative, **{f'delta_{k}': 0 for k in cumulative}, **(delta or {}),
           **report(fed, data, parts, alive)}
    if result is not None:
        row.update(fallback_rounds=result.details.get('fallback_rounds', 0),
                   condition_met=result.details.get('condition_met', np.nan))
    row.update(extra)
    return row


def _reference(data, parts, alive, seed, config, measure_payload):
    # Re-index the retained dataset while preserving the original row order and
    # client ids; this is the same seeded single-model FedAvg as fresh retraining.
    keep = np.flatnonzero(alive)
    mapping = np.full(len(alive), -1, dtype=np.int64)
    mapping[keep] = np.arange(len(keep))
    retained = {key: value[keep] for key, value in data['train'].items()}
    remaining = [mapping[p[alive[p]]] for p in parts]
    fed = ShardedUnlearner(retained, data['validation'], remaining, [list(range(len(parts)))],
                          seed, 'shard_full', config, measure_payload=measure_payload)
    return fed, fed.fit()


def validate_run_inputs(record, family, data, parts, config, seeds, shards, stream_config, reference_steps):
    """A startup record cannot label modified runtime arguments as protocol v3."""
    name = record.get('design_name')
    if name not in DESIGNS or record.get('design') != DESIGNS[name]:
        raise ValueError('startup record does not contain a supported frozen design')
    design = DESIGNS[name]
    expected_config = ShardConfig.from_train_config(record['train_config'],
                        min_clients_design=design['min_clients_design'], merge_below=design['merge_below'])
    checks = {'family': record.get('family') == family,
              'runner': record.get('runner') == RUNNER,
              'train_config': record.get('train_config') == (CONFIGS['standard'] if name == 'v3' else SMOKE_CONFIG),
              'train_config_name': record.get('train_config_name') == design['train_config'],
              'seeds': seeds == design['seeds'], 'shards': shards == design['shards'],
              'reference_steps': list(reference_steps) == design['reference_steps'],
              'stream_config': stream_config == STREAM == record.get('stream_config'),
              'component_config': config.to_json() == expected_config.to_json(),
              'data_content': data_content_sha256(data) == record.get('data_content_sha256') == DATA_SHA256,
              'client_rows': [len(p) for p in parts] == record.get('client_rows') == EXPECTED_CLIENT_ROWS,
              'machine': record.get('machine') == machine_info(config.device),
              'source_hashes': record.get('sha256_at_start') == source_hashes()}
    expected_parts = partition(data['train']['y'], design['partition_seed'], design['partition_alpha'], design['clients'])
    checks['partition'] = len(parts) == len(expected_parts) and all(np.array_equal(x, y) for x, y in zip(parts, expected_parts))
    if not all(checks.values()):
        raise ValueError(f'runtime inputs differ from startup record: {[k for k, v in checks.items() if not v]}')


def run_experiment(out, family, data, parts, config, seeds, shards=(4, 5), stream_config=None,
                   run_record=None, reference_steps=(5, 10, 15, 20), measure_payload=True):
    """Run one complete family and persist CSV/JSON artifacts.

    config accepts a ShardConfig or the shared train-config dictionary. Without a
    startup record this is labelled synthetic_smoke and can never confirm v3.
    Costs in steps.csv are cumulative after fit, plus per-operation delta_*.
    """
    if family not in FAMILIES:
        raise ValueError(f'unknown family {family!r}')
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    if (out/'steps.csv').exists() or (out/'exactness.csv').exists():
        raise FileExistsError(f'refusing to overwrite completed artifacts in {out}')
    cfg_dict = config if isinstance(config, dict) else {
        'fl': dict(config.fl), 'shard_fixed_rounds': config.shard_fixed_rounds, 'backfill_rounds': config.backfill_rounds}
    config = ShardConfig.from_train_config(config) if isinstance(config, dict) else config
    stream_config = copy.deepcopy(STREAM if stream_config is None else stream_config)
    parts = [np.asarray(p, dtype=np.int64) for p in parts]
    seeds, shards = list(seeds), list(shards)
    if len(set(seeds)) != len(seeds) or not seeds or len(set(shards)) != len(shards) or not shards:
        raise ValueError('seeds and shards must be nonempty and distinct')
    K = stream_config['K']
    if K not in reference_steps:
        raise ValueError('reference_steps must include the final request')
    started = time.perf_counter()
    record = copy.deepcopy(run_record)
    if record is not None:
        validate_run_inputs(record, family, data, parts, config, seeds, shards, stream_config, reference_steps)
    if record is None:
        record = {'design_name': 'synthetic_smoke', 'design': {**copy.deepcopy(DESIGNS['smoke_v3']),
                  'seeds': seeds, 'shards': shards, 'clients': len(parts), 'reference_steps': list(reference_steps)},
                  'train_config_name': 'smoke', 'train_config': cfg_dict, 'family': family, 'runner': RUNNER,
                  'data_content_sha256': data_content_sha256(data), 'sha256_at_start': source_hashes(),
                  'client_rows': [len(p) for p in parts], 'stream_config': stream_config,
                  'machine': machine_info(config.device), 'completed': False}
    record['component_config'] = config.to_json()
    record['reproducibility_checks'] = []
    record['reproducible'] = False
    write_json(out/'run.json', record)
    record['backend_warmup_seconds'] = warm_up(data['train']['X'].shape[1], config)
    streams = {str(S): materialize_stream(family, parts, assign(S, len(parts)), data['train'], stream_config) for S in shards}
    stream_record = {'format': 1, 'family': family, 'stream_config': stream_config,
                     'client_rows': [len(p) for p in parts], 'assignments': {str(S): assign(S, len(parts)) for S in shards},
                     'streams': {key: [{k: v for k, v in r.items() if k != 'f'} for r in value]
                                 for key, value in streams.items()}}
    write_json(out/'stream.json', stream_record)
    rows, checks, base_costs = [], [], []
    for S in shards:
        requests = streams[str(S)]
        for seed in seeds:
            base = ShardedUnlearner(data['train'], data['validation'], parts, assign(S, len(parts)), seed,
                                    'shard_full', config, measure_payload=measure_payload)
            fitted = base.fit()
            base_costs.append({'S': S, 'seed': seed, 'cost': fitted['cost']})
            reproduced = base.verify_bit_exact()
            record['reproducibility_checks'].append({'S': S, 'seed': seed, 'reproducible': reproduced['bit_exact'],
                                                     'per_shard': reproduced['per_shard']})
            write_json(out/'run.json', record)
            final_full = None
            for policy in POLICIES:
                fed = base.fork(policy if policy in ('shard_full', 'shard_fixed', 'utility_stop', 'drop_backfill') else 'shard_full')
                alive = np.ones(len(data['train']['y']), bool)
                cumulative = _zero_cost()
                context = {'family': family, 'S': S, 'model_shards': S, 'seed': seed, 'policy': policy}
                if policy.startswith('batch_full_'):
                    W = int(policy.rsplit('_', 1)[1])
                    times = batch_windows([r['time'] for r in requests], W)
                    groups = {}
                    for req, processed in zip(requests, times):
                        groups.setdefault(processed, []).append(req)
                    for processed, group in groups.items():
                        pending_rows, exposure = 0, 0.
                        for req in group:
                            alive[req['f']] = False
                            pending_rows += req['rows']
                            latency = processed - req['time']
                            exposure += req['rows']*latency
                            rows.append(_row(fed, data, parts, alive, context, req, 'request', cumulative,
                                             processed_at=processed, latency=latency, exposure=req['rows']*latency,
                                             pending_rows=pending_rows, exact=np.nan, structure_ok=np.nan))
                        result = fed.unlearn_batch([ForgetRequest(r['client'], r['f']) for r in group])
                        delta = _charge(cumulative, result)
                        end = {'step': group[-1]['step'], 'time': processed, 'rows': pending_rows}
                        rows.append(_row(fed, data, parts, alive, context, end, 'batch_end', cumulative, delta, result,
                                         exposure=exposure, batch_size=len(group)))
                else:
                    for req in requests:
                        alive[req['f']] = False
                        result = None if policy == 'none' else fed.unlearn(ForgetRequest(req['client'], req['f']))
                        delta = {} if result is None else _charge(cumulative, result)
                        rows.append(_row(fed, data, parts, alive, context, req, 'request', cumulative, delta, result,
                                         pending_rows=int((~alive).sum()) if policy == 'none' else 0))
                    if policy == 'drop_backfill':
                        result = fed.finish_backlog()
                        delta = _charge(cumulative, result)
                        end = {'step': K + 1, 'time': requests[-1]['time'], 'rows': 0}
                        rows.append(_row(fed, data, parts, alive, context, end, 'backlog_done', cumulative, delta, result))
                if policy != 'none':
                    verified = fed.verify_bit_exact()
                    checks.append({**context, 'bit_exact': verified['bit_exact'], 'shards_checked': verified['shards_checked'],
                                   'included_equals_active': verified['included_equals_active'],
                                   'structure_ok_end': verified['structure_ok'], 'exact': verified['exact'],
                                   'exactness_scope': 'conditional_on_round_count' if policy == 'utility_stop' else 'unconditional',
                                   'batch_matches_shard_full': weights_equal(fed, final_full) if policy.startswith('batch_full_') else np.nan})
                if policy == 'shard_full':
                    final_full = fed
            alive = np.ones(len(data['train']['y']), bool)
            cumulative = _zero_cost()
            for req in requests:
                alive[req['f']] = False
                if req['step'] not in reference_steps:
                    continue
                single, fitted = _reference(data, parts, alive, seed, config, measure_payload)
                delta = _charge(cumulative, fitted)
                context = {'family': family, 'S': S, 'model_shards': 1, 'seed': seed, 'policy': 'full_retrain_single'}
                rows.append(_row(single, data, parts, alive, context, req, 'reference', cumulative, delta))
            pd.DataFrame(rows).to_csv(out/'steps.csv', index=False)
            pd.DataFrame(checks).to_csv(out/'exactness.csv', index=False)
            print(f'{family} S={S} seed={seed} done; initial reproducible={reproduced["bit_exact"]}', flush=True)
    record.update(reproducible=all(c['reproducible'] for c in record['reproducibility_checks']),
                  completed=True, run_seconds=time.perf_counter() - started, base_fits=base_costs)
    write_json(out/'run.json', record)
    return record


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('out', type=Path)
    parser.add_argument('--attempt', type=int, choices=(1, 2), help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    # Checked before anything is created: the evaluator reads stream_<family>, and an
    # attempt directory belongs to the family directory that contains it.
    name, family = run_selection()
    family_dir = args.out.resolve() if args.attempt is None else args.out.resolve().parent
    if family_dir.name != f'stream_{family}':
        raise ValueError(f'output directory of family {family!r} must be named stream_{family}, got {family_dir.name!r}')
    args.out.mkdir(parents=True, exist_ok=False)
    if args.attempt is not None:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        _, design, cfg, data, parts, record = start_run(args.out)
        record['attempt'] = args.attempt
        run_experiment(args.out, record['family'], data, parts, cfg, design['seeds'], design['shards'],
                       run_record=record, reference_steps=design['reference_steps'])
        return 0
    seeds = DESIGNS[name]['seeds']
    print(f'{family}: design={name}, seeds {seeds[0]}..{seeds[-1]}, output {args.out}', flush=True)
    attempts = []
    for attempt in (1, 2):
        attempt_dir = args.out/f'attempt_{attempt}'
        completed = subprocess.run([sys.executable, str(Path(__file__).resolve()), str(attempt_dir), '--attempt', str(attempt)],
                                   env=os.environ.copy(), check=False)
        if completed.returncode:
            raise RuntimeError(f'v3 attempt {attempt} failed ({completed.returncode}); artifacts retained in {attempt_dir}')
        record = json.loads((attempt_dir/'run.json').read_text(encoding='utf-8'))
        attempts.append(record)
        if record['reproducible']:
            break
        print(f'{record["family"]}: initial fit not reproducible in attempt {attempt}', flush=True)
    for name in ('steps.csv', 'exactness.csv', 'stream.json'):
        shutil.copyfile(attempt_dir/name, args.out/name)
    record = {**record, 'attempts': attempts, 'selected_attempt': len(attempts)}
    write_json(args.out/'run.json', record)
    print(f'DONE {record["family"]}; reproducible={record["reproducible"]}; attempts={len(attempts)}', flush=True)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
