"""Fail-closed evaluation of the gates fixed in docs/shard_v3_plan.md.

Usage: python scripts/shard/evaluate_shard_v3.py RUN_ROOT [OUT_JSON]
RUN_ROOT contains stream_proportional, stream_spread, stream_concentrated and
stream_poisson_zipf. Smoke runs exercise the same checks but cannot confirm a claim.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
from scripts.shard.shard_v2 import DATA_SHA256, assign, noninferior, norm_rel, text_sha256
from scripts.shard.stream_v3 import FAMILIES, STREAM, backfill_plan, batch_windows, cost_plan, plan_stream
from scripts.shard.train_config import CONFIGS

HIGH_FPR, D_F1, D_FPR, ALPHA = .45, .01, .02, .05
POLICIES = ['none', 'shard_full', 'shard_fixed', 'utility_stop', 'drop_backfill', 'batch_full_2', 'batch_full_5']
EXACT = POLICIES[1:]
BATCHES = {'batch_full_2': 2, 'batch_full_5': 5}
METRICS = ['cal_macro_f1', 'cal_fpr', 'cal_recall_attack']
KEYS = ['family', 'S', 'seed', 'policy', 'kind', 'step']
SCOPES = {p: ('conditional_on_round_count' if p == 'utility_stop' else 'unconditional') for p in EXACT}


def _true(value):
    return isinstance(value, (bool, np.bool_)) and bool(value) or isinstance(value, str) and value == 'True'


def _false(value):
    return isinstance(value, (bool, np.bool_)) and not bool(value) or isinstance(value, str) and value == 'False'


def _close(a, b):
    try:
        return math.isfinite(float(a)) and math.isclose(float(a), float(b), rel_tol=1e-12, abs_tol=1e-12)
    except (ValueError, TypeError):
        return False


def _list(value):
    if isinstance(value, str):
        return json.loads(value)
    return value


def _clean(value):
    """Strict JSON output: unavailable metrics are null, never non-standard NaN."""
    if isinstance(value, dict):
        return {str(k): _clean(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_clean(v) for v in value]
    if isinstance(value, np.generic):
        return _clean(value.item())
    if isinstance(value, float) and not math.isfinite(value):
        return None
    return value


def _frame(rows, **where):
    selected = rows
    for name, value in where.items():
        selected = selected[selected[name] == value]
    return selected


def _paired(candidate, reference, seeds):
    """Require one finite row per seed; duplicates cannot silently count as pairs."""
    expected = sorted(seeds)
    for table in (candidate, reference):
        if sorted(table.seed.tolist()) != expected:
            raise ValueError('missing or duplicate paired seed')
        if not np.isfinite(table[METRICS[:2]].to_numpy(float)).all():
            raise ValueError('non-finite paired metric')
    a = candidate.set_index('seed').loc[expected]
    b = reference.set_index('seed').loc[expected]
    return a, b


def compare(candidate, reference, seeds):
    try:
        a, b = _paired(candidate, reference, seeds)
        f1 = noninferior(a.cal_macro_f1 - b.cal_macro_f1, D_F1, 'lower', ALPHA)
        fpr = noninferior(a.cal_fpr - b.cal_fpr, D_FPR, 'higher', ALPHA)
        high = a.index[a.cal_fpr > HIGH_FPR].tolist()
        return {'ok': f1['ok'] and fpr['ok'] and not high, 'dF1_mean': f1['mean'],
                'dF1_lower95': f1['bound'], 'dFPR_mean': fpr['mean'], 'dFPR_upper95': fpr['bound'],
                'F1_mean': a.cal_macro_f1.mean(), 'FPR_max': a.cal_fpr.max(), 'high_fpr_seeds': high}
    except (KeyError, ValueError, TypeError) as exc:
        return {'ok': False, 'reason': str(exc)}


def expected_keys(streams, seeds, shards):
    wanted = set()
    for family in FAMILIES:
        for S in shards:
            st = streams[family]['streams'][str(S)]
            K = len(st)
            for seed in seeds:
                for policy in POLICIES:
                    for step in range(1, K + 1):
                        wanted.add((family, S, seed, policy, 'request', step))
                wanted.add((family, S, seed, 'drop_backfill', 'backlog_done', K + 1))
                for step in range(5, K + 1, 5):
                    wanted.add((family, S, seed, 'full_retrain_single', 'reference', step))
                for policy, W in BATCHES.items():
                    ends = {}
                    for request, at in zip(st, batch_windows([r['time'] for r in st], W)):
                        ends[at] = request['step']
                    for step in ends.values():
                        wanted.add((family, S, seed, policy, 'batch_end', step))
    return wanted


def coverage(steps, exactness, streams, seeds, shards):
    expected = expected_keys(streams, seeds, shards)
    actual = list(steps[KEYS].itertuples(index=False, name=None))
    xkeys = ['family', 'S', 'seed', 'policy']
    want_x = {(f, S, s, p) for f in FAMILIES for S in shards for s in seeds for p in EXACT}
    got_x = list(exactness[xkeys].itertuples(index=False, name=None))
    problems = {}
    for label, got, want in [('steps', actual, expected), ('exactness', got_x, want_x)]:
        missing, extra = want - set(got), set(got) - want
        duplicates = len(got) - len(set(got))
        if missing or extra or duplicates:
            problems[label] = {'missing': sorted(missing), 'unexpected': sorted(extra), 'duplicate_count': duplicates}
    bad_metrics = []
    for row in steps.to_dict('records'):
        values = np.asarray([row[m] for m in METRICS], dtype=float)
        unavailable = _true(row['unavailable'])
        valid_flag = unavailable or _false(row['unavailable'])
        valid = (row['policy'] == 'drop_backfill' and row['kind'] == 'request' and row['shards_included'] == 0
                 and np.isnan(values).all()) if unavailable else (row['shards_included'] > 0 and
                 np.isfinite(values).all() and ((values >= 0) & (values <= 1)).all())
        valid = valid and row['model_shards'] == (1 if row['policy'] == 'full_retrain_single' else row['S'])
        if not valid or not valid_flag:
            bad_metrics.append([row[k] for k in KEYS])
    if bad_metrics:
        problems['invalid_metrics_or_availability'] = bad_metrics
    return {'pass': not problems, 'failures': problems, 'expected_steps': len(expected), 'expected_exactness': len(want_x)}


def g1(steps, exactness, covered):
    checked = steps[steps.policy.isin(EXACT) & (~steps.policy.isin(BATCHES) | (steps.kind == 'batch_end'))]
    failures = []
    for row in checked.to_dict('records'):
        if not (_true(row['exact']) and _true(row['structure_ok']) and row['exactness_scope'] == SCOPES[row['policy']]):
            failures.append([row[k] for k in KEYS])
    end_failures = []
    for row in exactness.to_dict('records'):
        fields = ['bit_exact', 'included_equals_active', 'structure_ok_end', 'exact']
        ok = all(_true(row[f]) for f in fields) and row['shards_checked'] == row['S']
        ok = ok and row['exactness_scope'] == SCOPES.get(row['policy'])
        if row['policy'] in BATCHES:
            ok = ok and _true(row['batch_matches_shard_full'])
        if not ok:
            end_failures.append([row[k] for k in ('family', 'S', 'seed', 'policy')])
    return {'pass': covered['pass'] and not failures and not end_failures, 'coverage_complete': covered['pass'],
            'step_failures': failures, 'final_failures': end_failures, 'exactness_scope': SCOPES}


def g2(steps, streams, seeds, shards, cfg, stream_cfg):
    failures, comparisons = [], []
    rounds = cfg['fl']['rounds']
    for family in FAMILIES:
        document = streams[family]
        n = document['client_rows']
        if document.get('family') != family or document.get('stream_config') != stream_cfg:
            failures.append({'family': family, 'check': 'stream_metadata'})
        for S in shards:
            A = assign(S, len(n))
            expected = plan_stream(family, n, A, stream_cfg)
            st = document['streams'].get(str(S), [])
            canonical = [{key: r.get(key) for key in expected[0]} for r in st]
            if document['assignments'].get(str(S)) != A or canonical != expected:
                failures.append({'family': family, 'S': S, 'check': 'stream_plan'})
            subset = _frame(steps, family=family, S=S)
            for seed in seeds:
                current = _frame(subset, seed=seed)
                for policy in POLICIES:
                    req = _frame(current, policy=policy, kind='request').sort_values('step')
                    if len(req) != len(expected):
                        failures.append({'family': family, 'S': S, 'seed': seed, 'policy': policy, 'check': 'request_count'})
                        continue
                    for row, planned in zip(req.to_dict('records'), expected):
                        ok = all(_close(row[key], planned[key]) for key in ('step', 'time', 'client', 'shard', 'requested', 'rows'))
                        saved = st[int(planned['step']) - 1] if len(st) == len(expected) else {}
                        digest = saved.get('forget_request_hash')
                        ok = ok and isinstance(digest, str) and len(digest) == 64 and row.get('forget_request_hash') == digest
                        if not ok:
                            failures.append({'family': family, 'S': S, 'seed': seed, 'policy': policy,
                                             'step': planned['step'], 'check': 'executed_request'})
                for policy, W in {'shard_full': 0, **BATCHES}.items():
                    planned = cost_plan(expected, A, n, rounds, W)
                    pol = _frame(current, policy=policy)
                    proc = batch_windows([r['time'] for r in expected], W)
                    groups = {}
                    for request, at in zip(expected, proc):
                        groups.setdefault(at, []).append(request)
                    cost, total_rounds = 0, 0
                    left = np.asarray(n, dtype=int).copy()
                    valid = True
                    for at, batch in groups.items():
                        for request in batch:
                            left[request['client']] -= request['rows']
                        touched = sorted({r['shard'] for r in batch})
                        delta = sum(int(left[A[j]].sum()) for j in touched) * rounds
                        cost += delta
                        total_rounds += len(touched) * rounds
                        end = _frame(pol, kind='batch_end' if W else 'request', step=batch[-1]['step'])
                        if len(end) != 1:
                            valid = False
                            continue
                        row = end.iloc[0]
                        valid = valid and all(_close(row[k], v) for k, v in {
                            'rows_x_rounds': cost, 'delta_rows_x_rounds': delta,
                            'rounds': total_rounds, 'delta_rounds': len(touched) * rounds,
                            'processed_at': at, 'pending_rows': 0}.items())
                        if W:
                            exposure = sum(r['rows'] * (at-r['time']) for r in batch)
                            valid = valid and _close(row.exposure, exposure) and _close(row.time, at)
                            valid = valid and _close(row.rows, sum(r['rows'] for r in batch))
                    requests = _frame(pol, kind='request').sort_values('step')
                    if len(requests) != len(expected):
                        valid = False
                    else:
                        pending, last = 0, None
                        before_cost, before_rounds = 0, 0
                        for row, request, at in zip(requests.to_dict('records'), expected, proc):
                            if at != last:
                                pending = 0
                                last = at
                                previous = pol[(pol.kind == 'batch_end') & (pol.time < request['time'])].sort_values('time')
                                if len(previous):
                                    before_cost, before_rounds = previous.iloc[-1][['rows_x_rounds', 'rounds']]
                            pending += request['rows']
                            valid = valid and all(_close(row[k], v) for k, v in {
                                'processed_at': at, 'latency': at-request['time'],
                                'exposure': request['rows']*(at-request['time']), 'pending_rows': pending if W else 0}.items())
                            if W:
                                valid = valid and all(_close(row[k], v) for k, v in {
                                    'delta_rows_x_rounds': 0, 'delta_rounds': 0,
                                    'rows_x_rounds': before_cost, 'rounds': before_rounds}.items())
                    valid = valid and cost == planned['shard_rows_x_rounds'] and total_rounds == planned['shard_retrains']*rounds
                    comparisons.append({'family': family, 'S': S, 'seed': seed, 'policy': policy, 'ok': bool(valid), **planned})
                expected_inc = backfill_plan([r['shard'] for r in expected], S, rounds, cfg['backfill_rounds'])
                backfill = _frame(current, policy='drop_backfill', kind='request').sort_values('step')
                valid = len(backfill) == len(expected_inc)
                for row, included in zip(backfill.to_dict('records'), expected_inc):
                    valid = valid and row['shards_included'] == len(included) and _list(row['included_shards']) == included
                    valid = valid and (_true(row['unavailable']) if not included else _false(row['unavailable']))
                if not valid:
                    failures.append({'family': family, 'S': S, 'seed': seed, 'check': 'backfill_plan'})
    return {'pass': not failures and all(c['ok'] for c in comparisons), 'failures': failures, 'comparisons': comparisons}


def g3_g4(steps, seeds, shards, K, cheaper=False):
    comparisons = []
    for family in FAMILIES:
        for S in shards:
            group = _frame(steps, family=family, S=S, step=K)
            ref = _frame(group, policy='shard_full' if cheaper else 'full_retrain_single', kind='request' if cheaper else 'reference')
            for policy in (['shard_fixed', 'utility_stop'] if cheaper else ['shard_full']):
                candidate = _frame(group, policy=policy, kind='request')
                comparisons.append({'family': family, 'S': S, 'policy': policy, **compare(candidate, ref, seeds)})
    return {'pass': all(c['ok'] for c in comparisons), 'comparisons': comparisons}


def g5(steps, seeds, shards, K):
    comparisons = []
    for family in FAMILIES:
        for S in shards:
            group = _frame(steps, family=family, S=S, kind='request')
            differences, unavailable = [], []
            reason = None
            for seed in seeds:
                a = _frame(group, policy='drop_backfill', seed=seed).sort_values('step')
                b = _frame(group, policy='shard_full', seed=seed).sort_values('step')
                if a.step.tolist() != list(range(1, K+1)) or b.step.tolist() != list(range(1, K+1)):
                    reason = 'missing or duplicate paired step'
                    continue
                if a.unavailable.map(_true).any():
                    unavailable.append(seed)
                    continue
                if not np.isfinite(a[METRICS[:2]].to_numpy(float)).all() or not np.isfinite(b[METRICS[:2]].to_numpy(float)).all():
                    reason = 'non-finite paired metric'
                    continue
                differences.append({'seed': seed, 'dF1': float(np.min(a.cal_macro_f1.to_numpy()-b.cal_macro_f1.to_numpy())),
                                    'dFPR': float(np.max(a.cal_fpr.to_numpy()-b.cal_fpr.to_numpy()))})
            result = {'family': family, 'S': S, 'ok': False, 'unavailable_seeds': unavailable, 'worst_by_seed': differences}
            if reason:
                result['reason'] = reason
            if not reason and not unavailable and len(differences) == len(seeds):
                f1 = noninferior([d['dF1'] for d in differences], D_F1, 'lower', ALPHA)
                fpr = noninferior([d['dFPR'] for d in differences], D_FPR, 'higher', ALPHA)
                result.update(ok=f1['ok'] and fpr['ok'], dF1_mean=f1['mean'], dF1_lower95=f1['bound'],
                              dFPR_mean=fpr['mean'], dFPR_upper95=fpr['bound'])
            comparisons.append(result)
    return {'pass': all(c['ok'] for c in comparisons), 'comparisons': comparisons}


def g6(steps, seeds, shards, K):
    utility = _frame(steps, policy='utility_stop', kind='request')
    expected = {(f, S, s, k) for f in FAMILIES for S in shards for s in seeds for k in range(1, K+1)}
    actual = list(utility[['family', 'S', 'seed', 'step']].itertuples(index=False, name=None))
    bad = utility[(~np.isfinite(utility.cal_fpr.to_numpy(float))) | (utility.cal_fpr > HIGH_FPR)]
    return {'pass': len(actual) == len(expected) and set(actual) == expected and bad.empty,
            'FPR_max': utility.cal_fpr.max(), 'high_fpr': bad[['family', 'S', 'seed', 'step', 'cal_fpr']].to_dict('records')}


def integrity(records, streams):
    from scripts.shard.shard_v3 import DESIGN, EXPECTED_CLIENT_ROWS, RUNNER, TRACKED_V3
    from src.fu.shard_exact import ShardConfig
    result = {}
    machines = []
    for family in FAMILIES:
        rec = records[family]
        hashes = rec.get('sha256_at_start', {})
        mismatch = sorted(set(TRACKED_V3) ^ set(hashes))
        mismatch += [f for f in TRACKED_V3 if f in hashes and (not (ROOT/f).exists() or text_sha256(ROOT/f) != hashes[f])]
        checks = rec.get('reproducibility_checks', [])
        pairs = [(c.get('S'), c.get('seed')) for c in checks]
        wanted = {(S, s) for S in DESIGN['shards'] for s in DESIGN['seeds']}
        machine = rec.get('machine')
        machines.append(machine)
        machine_keys = {'hostname', 'platform', 'machine', 'processor', 'python', 'numpy', 'pandas',
                        'scipy', 'sklearn', 'torch', 'threads', 'device'}
        attempts = rec.get('attempts', [])
        attempt_ok = isinstance(attempts, list) and len(attempts) in (1, 2) and rec.get('selected_attempt') == len(attempts)
        if attempt_ok:
            same_fields = ['design_name', 'design', 'family', 'train_config_name', 'train_config', 'runner',
                           'data_content_sha256', 'sha256_at_start', 'client_rows', 'stream_config', 'machine',
                           'component_config']
            attempt_ok = all(all(a.get(k) == rec.get(k) for k in same_fields) and a.get('attempt') == i + 1
                             and _true(a.get('completed')) for i, a in enumerate(attempts))
            attempt_ok = attempt_ok and attempts[-1].get('reproducibility_checks') == checks and _true(attempts[-1].get('reproducible'))
            if len(attempts) == 2:
                attempt_ok = attempt_ok and _false(attempts[0].get('reproducible'))
                attempt_ok = attempt_ok and any(_false(c.get('reproducible')) for c in attempts[0].get('reproducibility_checks', []))
        tests = {'design_v3': rec.get('design_name') == 'v3' and rec.get('design') == DESIGN,
                 'config_standard': rec.get('train_config_name') == 'standard' and rec.get('train_config') == CONFIGS['standard'],
                 'component_config': rec.get('component_config') == ShardConfig.from_train_config(CONFIGS['standard']).to_json(),
                 'stream_config': rec.get('stream_config') == STREAM,
                 'runner': norm_rel(ROOT/rec.get('runner', '')) == RUNNER,
                 'family': rec.get('family') == family and streams[family].get('family') == family,
                 'data_content': rec.get('data_content_sha256') == DATA_SHA256,
                 'client_rows': streams[family].get('client_rows') == rec.get('client_rows') == list(EXPECTED_CLIENT_ROWS),
                 'tracked_sha256': not mismatch, 'completed': _true(rec.get('completed')),
                 'machine_recorded': isinstance(machine, dict) and machine_keys <= set(machine) and machine.get('device') == 'cpu'
                                     and type(machine.get('threads')) is int and machine['threads'] > 0,
                 'attempt_history': bool(attempt_ok),
                 'reproducible': _true(rec.get('reproducible')) and len(pairs) == len(wanted) and set(pairs) == wanted
                                 and all(_true(c.get('reproducible')) for c in checks)}
        result[family] = {'ok': all(tests.values()), 'checks': tests, 'sha256_mismatch': mismatch}
    same_machine = bool(machines[0]) and all(m == machines[0] for m in machines)
    return {'pass': all(r['ok'] for r in result.values()) and same_machine,
            'same_machine_and_versions': same_machine, 'families': result}


def descriptive(steps):
    summaries = []
    for (family, S, seed, policy), group in steps.groupby(['family', 'S', 'seed', 'policy'], sort=True):
        requests = group[group.kind == 'request']
        end = group.sort_values(['time', 'step']).iloc[-1]
        summaries.append({'family': family, 'S': S, 'seed': seed, 'policy': policy,
                          **{c: end[c] for c in ('rows_x_rounds', 'rounds', 'bytes', 'seconds')},
                          'unavailable_steps': int(requests.unavailable.map(_true).sum()),
                          'fallback_rounds': float(group.fallback_rounds.fillna(0).sum()),
                          'condition_unmet_steps': int(requests.condition_met.map(_false).sum()) if policy == 'utility_stop' else 0,
                          'latency_mean': requests.latency.mean() if policy in BATCHES else 0.,
                          'latency_max': requests.latency.max() if policy in BATCHES else 0.,
                          'exposure': requests.exposure.sum() if policy in BATCHES else 0.})
    return {'policy_costs_and_latency': summaries,
            'single_model_checkpoints': steps[steps.policy == 'full_retrain_single'][['family', 'S', 'seed', 'step']+METRICS].to_dict('records'),
            'batch_intermediate_utility': steps[steps.policy.isin(BATCHES) & (steps.kind == 'request')][KEYS+METRICS+['pending_rows']].to_dict('records')}


def evaluate_frames(steps, exactness, streams, records):
    """Evaluate in-memory artifacts; also used by the runner's synthetic smoke tests."""
    from scripts.shard.shard_v3 import DESIGN, DESIGNS
    names = {r.get('design_name') for r in records.values()}
    name = next(iter(names)) if len(names) == 1 else 'v3'
    design = records[FAMILIES[0]]['design'] if name == 'synthetic_smoke' else DESIGNS.get(name, DESIGN)
    seeds, shards = design['seeds'], design['shards']
    stream_cfg = records[FAMILIES[0]].get('stream_config', STREAM)
    cfg = records[FAMILIES[0]]['train_config']
    K = stream_cfg['K']
    covered = coverage(steps, exactness, streams, seeds, shards)
    integ = integrity(records, streams)
    refs = _frame(steps, policy='full_retrain_single', kind='reference')
    bad_refs = refs[(~np.isfinite(refs.cal_fpr.to_numpy(float))) | (refs.cal_fpr > HIGH_FPR)]
    gates = {'G1': g1(steps, exactness, covered), 'G2': g2(steps, streams, seeds, shards, cfg, stream_cfg),
             'G3': g3_g4(steps, seeds, shards, K), 'G4': g3_g4(steps, seeds, shards, K, cheaper=True),
             'G5': g5(steps, seeds, shards, K), 'G6': g6(steps, seeds, shards, K)}
    pre = integ['pass'] and covered['pass'] and bad_refs.empty
    utility_rows = [c for c in gates['G4']['comparisons'] if c['policy'] == 'utility_stop']
    claims = {'exact_unlearning': gates['G1']['pass'], 'precomputed_cost_and_latency': gates['G2']['pass'],
              'sharding_keeps_single_model_utility': gates['G3']['pass'], 'cheaper_policies_match_shard_full': gates['G4']['pass'],
              'drop_backfill_safe_at_every_step': gates['G5']['pass'],
              'utility_stop_safe': gates['G6']['pass'] and all(c['ok'] for c in utility_rows)}
    return _clean({'confirmed_all': pre and all(g['pass'] for g in gates.values()),
                   'P': {'pass': pre, 'integrity_ok': integ['pass'], 'coverage_complete': covered['pass'],
                         'reference_high_fpr': bad_refs[['family', 'S', 'seed', 'step', 'cal_fpr']].to_dict('records')},
                   'claims': {k: pre and bool(v) for k, v in claims.items()}, 'design_name': name, 'seeds': seeds,
                   'thresholds': {'HIGH_FPR': HIGH_FPR, 'D_F1': D_F1, 'D_FPR': D_FPR, 'alpha_one_sided': ALPHA},
                   'gates': gates, 'coverage': covered, 'integrity': integ, 'descriptive': descriptive(steps)})


def evaluate(root):
    root = Path(root)
    try:
        steps, exactness, streams, records = [], [], {}, {}
        for family in FAMILIES:
            directory = root/f'stream_{family}'
            steps.append(pd.read_csv(directory/'steps.csv'))
            exactness.append(pd.read_csv(directory/'exactness.csv'))
            streams[family] = json.loads((directory/'stream.json').read_text(encoding='utf-8'))
            records[family] = json.loads((directory/'run.json').read_text(encoding='utf-8'))
        return evaluate_frames(pd.concat(steps, ignore_index=True), pd.concat(exactness, ignore_index=True), streams, records)
    except (OSError, ValueError, TypeError, KeyError, IndexError) as exc:
        return {'confirmed_all': False, 'P': {'pass': False}, 'input_error': f'{type(exc).__name__}: {exc}', 'claims': {}}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', nargs='?', type=Path)
    args = parser.parse_args(argv)
    result = evaluate(args.root)
    print(f"P: {'PASS' if result['P']['pass'] else 'FAIL'}")
    for key, gate in result.get('gates', {}).items():
        print(f"{key}: {'PASS' if gate['pass'] else 'FAIL'}")
    if 'input_error' in result:
        print(result['input_error'])
    print(f"CONFIRMED (all gates): {result['confirmed_all']}")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(result, indent=2, allow_nan=False), encoding='utf-8')
    return 2 if 'input_error' in result else 0


if __name__ == '__main__':
    raise SystemExit(main())
