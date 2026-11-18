"""Post-run checks of frozen logs; no fitting, tuning, or changes to run outputs.

Run from the repository root:
    python results/dt_predictive_confirmation/audit_replay.py
Writes only supplemental_audit.json and aggregate_metrics.json beside this file.
This audit was written after the run; it is not part of the frozen algorithm.
"""
import hashlib
import itertools
import json
import math
import csv
from pathlib import Path

import numpy as np

OUT = Path(__file__).resolve().parent
ROOT = OUT.parents[1]


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def close(actual, expected):
    assert math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-8), (actual, expected)


def weights(losses, cfg):
    scores = np.exp(-cfg['expert_eta'] * (losses - losses.min()))
    return (1-cfg['expert_floor']) * scores/scores.sum() + cfg['expert_floor']/len(scores)


lock = read(OUT/'protocol_lock.json')
cfg = lock['config']
hash_counts = {}
for name, manifest in (
    ('predictive', lock),
    ('previous_online', read(ROOT/'results/dt_online_confirmation/protocol_lock.json')),
):
    for path, expected in manifest['source_sha256'].items():
        assert hashlib.sha256((ROOT/path).read_bytes()).hexdigest() == expected, path
    hash_counts[name] = len(manifest['source_sha256'])

expected_blocks = {
    f'r{repeat}_{family}_{env}'
    for repeat, family, env in itertools.product(
        range(len(cfg['seeds'])), cfg['families'], cfg['environments'])
}
assert {p.parent.name for p in OUT.glob('*/run.json')} == expected_blocks
all_events = {name: [] for name in cfg['predictors']}
all_summaries = {name: [] for name in cfg['predictors']}
replayed = {name: 0 for name in ('dt_queue', 'rows_queue')}
max_weight_error = 0.
streams = 0
fit_round_seconds = []
for block in sorted(expected_blocks):
    directory = OUT/block
    run = read(directory/'run.json')
    rows = read(directory/'complete.json')['summaries']
    assert sorted(r['predictor'] for r in rows) == sorted(cfg['predictors'])
    assert {p.parent.name for p in directory.glob('*/final.json')} == set(cfg['predictors'])
    assert len({r['alive_rows'] for r in rows}) == 1
    requests = {r['step']: r for r in run['requests']}
    assert len(requests) == cfg['requests']
    with (directory/'fit_round_costs.csv').open(newline='', encoding='utf-8') as handle:
        fit_round_seconds.extend(float(r['wall_seconds']) for r in csv.DictReader(handle))
    close(run['budget_seconds'], run['reference_full_seconds']*cfg['budget_scale'])
    close(run['deadline_seconds'], run['reference_full_seconds']*cfg['deadline_scale'])
    for row in rows:
        name = row['predictor']
        events = [json.loads(line) for line in (directory/name/'events.jsonl').read_text().splitlines()]
        final = read(directory/name/'final.json')
        for key, value in final['summary'].items():
            assert row[key] == value, (block, name, key)
        exact = final['verify_bit_exact']
        assert all(exact[key] for key in ('bit_exact', 'exact', 'structure_ok', 'included_equals_active'))
        assert exact['shards_checked'] == cfg['shards'] and all(exact['per_shard'].values())
        assert sorted(s for event in events for s in event['steps']) == sorted(requests)
        losses = np.zeros(4 if name == 'dt_queue' else 1)
        prior_ready, served = 0., set()
        last_rounds = {str(s): 20 for s in range(cfg['shards'])}
        for i, event in enumerate(events, 1):
            n = len(event['steps'])
            assert event['event'] == i and event['requests'] == n
            assert all(len(event[key]) == n for key in ('arrivals', 'waiting_seconds', 'latency_seconds', 'lateness_seconds'))
            assert event['accepted_requests'] == n and event['rejected_requests'] == 0
            assert event['started_seconds'] >= prior_ready
            prior_ready = event['model_ready_seconds']
            close(event['service_seconds'], event['model_ready_seconds']-event['started_seconds'])
            close(event['deadline_seconds'], run['deadline_seconds'])
            close(event['pooled_budget_seconds'], n*run['budget_seconds'])
            assert event['round_over_budget'] == (event['round_seconds'] > event['pooled_budget_seconds'])
            assert event['service_over_budget'] == (event['service_seconds'] > event['pooled_budget_seconds'])
            assert event['exact'] and event['structure_ok']
            for step, arrival, waiting, latency, lateness in zip(
                event['steps'], event['arrivals'], event['waiting_seconds'], event['latency_seconds'], event['lateness_seconds']
            ):
                request = requests[step]
                assert step not in served and request['shard'] == event['shard']
                assert arrival == request['time'] and arrival <= event['started_seconds']
                close(waiting, event['started_seconds']-arrival)
                close(latency, event['model_ready_seconds']-arrival)
                close(lateness, max(0., latency-run['deadline_seconds']))
            assert event['deadline_misses'] == sum(t > run['deadline_seconds'] for t in event['latency_seconds'])
            last_rounds[str(event['shard'])] = 20 if event['policy'] == 'shard_full' else 10
            if name != 'rows_legacy':
                prediction = event['forecast_details']
                expected = weights(losses, cfg)
                error = float(np.max(np.abs(expected-np.asarray(prediction['weights']))))
                max_weight_error = max(max_weight_error, error)
                assert error < 1e-12
                close(prediction['mean_round'], expected @ np.asarray(prediction['experts']))
                close(prediction['upper_round'], prediction['mean_round']*prediction['margin'])
                close(prediction['service_upper'], prediction['upper_round']+prediction['overhead_seconds'])
                close(prediction['service_upper'], event['forecast_service_seconds'])
                choice = event['rollout']
                assert choice['shard'] == event['shard'] == choice['order'][0]
                assert choice['policy'] == event['policy'] == choice['policies'][0]
                assert 1 <= choice['plans_evaluated'] <= 384
                for shard in choice['order']:
                    assert any(r['shard'] == shard and r['time'] <= event['started_seconds'] and s not in served
                               for s, r in requests.items())
                # Only after checking this decision, consume the result of its execution.
                errors = np.abs(np.log(max(event['round_seconds'], 1e-12)/np.maximum(prediction['experts'], 1e-12)))
                alpha = cfg['expert_loss_alpha']
                losses = (1-alpha)*losses + alpha*errors
                replayed[name] += 1
            served.update(event['steps'])
        assert final['shard_rounds'] == last_rounds
        if name != 'rows_legacy':
            np.testing.assert_allclose(final['final_expert_weights'], weights(losses, cfg), rtol=0, atol=1e-12)
            close(row['plan_seconds_total'], sum(e['plan_seconds'] for e in events))
            close(row['sync_seconds_total'], sum(e['sync_seconds'] for e in events))
        joint = sum(sum(t <= run['deadline_seconds'] for t in e['latency_seconds'])
                    for e in events if e['round_seconds'] <= e['pooled_budget_seconds'])
        assert joint == row['joint_success_requests']
        all_events[name].extend(events)
        all_summaries[name].append(row)
        streams += 1

aggregates = {}
for name, events in all_events.items():
    rows = all_summaries[name]
    latencies = [t for e in events for t in e['latency_seconds']]
    n = len(latencies)
    service = sum(e['service_seconds'] for e in events)
    planning = sum(e['plan_seconds'] for e in events)
    syncing = sum(e['sync_seconds'] for e in events)
    aggregates[name] = dict(
        streams=len(rows), requests=n, events=len(events),
        joint_success=sum(r['joint_success_requests'] for r in rows),
        joint_success_percent=100*sum(r['joint_success_requests'] for r in rows)/n,
        deadline_misses=sum(e['deadline_misses'] for e in events),
        within_round_budget_requests=sum(e['requests'] for e in events if not e['round_over_budget']),
        latency_mean_seconds=float(np.mean(latencies)),
        latency_p95_seconds=float(np.quantile(latencies, .95)),
        service_seconds=service, full_requests=sum(e['requests'] for e in events if e['policy'] == 'shard_full'),
        validation_macro_f1_mean=float(np.mean([r['validation_macro_f1'] for r in rows])),
        validation_fpr_mean=float(np.mean([r['validation_fpr'] for r in rows])),
        recorded_plan_seconds=planning, recorded_sync_seconds=syncing,
        recorded_control_ms_per_event=1000*(planning+syncing)/len(events),
        recorded_control_percent_of_service=100*(planning+syncing)/service,
        planning_includes_wait_iterations=name != 'rows_legacy',
    )

result = dict(
    status='passed', blocks=len(expected_blocks), streams=streams,
    requests=sum(a['requests'] for a in aggregates.values()), bit_exact_streams=streams,
    hash_files_checked=hash_counts, weight_replay_events=replayed,
    max_weight_abs_error=max_weight_error,
    weight_replay_scope='Pre-execution weights from earlier executed-event errors and final weights; '
                       'does not reconstruct internal client-cost predictor states or prove statistical superiority.',
    causal_arrivals_and_rollouts=True, one_worker=True, no_missing_duplicate_or_dropped_requests=True,
    cost_flags_and_latency_audited=True, identical_alive_rows_within_blocks=True,
    final_policy_rounds_match=True,
    fit_round_seconds_range=[min(fit_round_seconds), max(fit_round_seconds)],
)
assert max(fit_round_seconds) <= cfg['calibration_max_round_seconds']
for filename, data in [('supplemental_audit.json', result), ('aggregate_metrics.json', aggregates)]:
    (OUT/filename).write_text(json.dumps(data, indent=2, allow_nan=False)+'\n', encoding='utf-8')
print(json.dumps(result, indent=2))
print(json.dumps(aggregates, indent=2))
