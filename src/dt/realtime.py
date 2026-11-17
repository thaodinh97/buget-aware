"""Single-worker, open-loop arrivals and measured model-ready deadlines.

The harness releases a request at its scheduled perf_counter timestamp, including
while training blocks the worker. Only arrived requests are exposed to forecasts.
Real processing time advances the same clock used by arrivals and deadlines.
The injectable clock is exclusively for deterministic scheduler tests.
"""
import math
import time

from src.dt import planner


class OnlineTwin:
    def __init__(self, name, fit_rounds, fit_clients, counts, alpha=.3, q=.9, window=10):
        profile = planner.rates(fit_rounds, fit_clients)
        self.predictor = planner.make_predictor(name, profile, fit_rounds, alpha)
        initial = planner.initial_ratios(self.predictor, fit_rounds, fit_clients)
        self.margin = planner.Margin(q, initial, window)
        self.counts = counts

    def forecast(self, shard, batch, rounds):
        return self.predictor.forecast(shard, self.counts(shard, batch), rounds)

    def factor(self):
        return self.margin.factor()

    def observe(self, shard, result, forecast):
        self.predictor.update(shard, result['round_costs'], result['client_costs'])
        self.margin.observe(result['round_seconds'] / forecast)


def run(requests, twin, execute, budget, deadline, full_rounds=20, fixed_rounds=10,
        padding=.05, clock=time.perf_counter, sleep=time.sleep, on_event=None):
    """Run all requests to completion, never receive free feedback from rejected work.

    Each arrived request contributes `budget` round-seconds to its shard batch.
    Run full now if forecast fits both pooled budget and the oldest deadline.
    Otherwise wait for more arrivals until the predicted latest start for fixed;
    then run fixed, even if budget/deadline is infeasible, and report that fact.
    Among eligible batches choose the oldest deadline. A batch cannot include
    arrivals during its own execution. execute returns once the model is usable.
    """
    if budget <= 0 or deadline <= 0 or padding < 0:
        raise ValueError('positive budget/deadline and nonnegative padding required')
    if any(r['time'] < 0 for r in requests) or any(a['time'] > b['time'] for a, b in zip(requests, requests[1:])):
        raise ValueError('arrival times must be nonnegative and sorted')
    origin, i, queues, events = clock(), 0, {}, []
    while i < len(requests) or any(queues.values()):
        now = clock() - origin
        while i < len(requests) and requests[i]['time'] <= now:
            r = requests[i]
            queues.setdefault(r['shard'], []).append(r)
            i += 1
        plan_start = clock()
        choices, wakeups = [], []
        for shard, batch in sorted(queues.items()):
            if not batch:
                continue
            f_full = twin.forecast(shard, batch, full_rounds)
            f_fixed = twin.forecast(shard, batch, fixed_rounds)
            factor = twin.factor()
            if not all(math.isfinite(x) and x > 0 for x in (f_full, f_fixed, factor)):
                raise ValueError('forecasts and margin must be finite and positive')
            due, pooled = batch[0]['time'] + deadline, len(batch) * budget
            current = clock() - origin
            cutoff = due - f_fixed * factor - padding
            if f_full * factor <= pooled and current + f_full * factor + padding <= due:
                choices.append((due, shard, 'shard_full', f_full, factor, False))
            elif current >= cutoff:
                choices.append((due, shard, 'shard_fixed', f_fixed, factor, f_fixed * factor > pooled))
            else:
                wakeups.append(cutoff)
        plan_seconds = clock() - plan_start
        if not choices:
            if i < len(requests):
                wakeups.append(requests[i]['time'])
            if not wakeups:
                raise RuntimeError('nonempty workload without an eligible event or wakeup')
            sleep(max(0., min(wakeups) - (clock() - origin)))
            continue
        due, shard, policy, predicted, factor, forced = min(choices)
        batch, queues[shard] = queues[shard], []
        started = clock() - origin
        result = execute(batch, policy, started)
        ready = clock() - origin
        # Completion is recorded BEFORE telemetry processing and file writes.
        # Their actual time still delays the next task on the same worker.
        observed = clock()
        twin.observe(shard, result, predicted)
        sync_seconds = clock() - observed
        latency = [ready - r['time'] for r in batch]
        waits = [started - r['time'] for r in batch]
        pooled = budget * len(batch)
        event = {k: v for k, v in result.items() if k not in ('round_costs', 'client_costs')}
        event.update(event=len(events) + 1, shard=shard, steps=[r['step'] for r in batch],
                     arrivals=[r['time'] for r in batch], requests=len(batch), policy=policy,
                     started_seconds=started, model_ready_seconds=ready, service_seconds=ready-started,
                     waiting_seconds=waits, latency_seconds=latency, deadline_seconds=deadline,
                     lateness_seconds=[max(0., x-deadline) for x in latency],
                     deadline_misses=sum(x > deadline for x in latency),
                     pooled_budget_seconds=pooled, forecast_round_seconds=predicted, margin_factor=factor,
                     forced_over_budget=forced, round_over_budget=result['round_seconds'] > pooled,
                     service_over_budget=ready-started > pooled,
                     round_overrun_seconds=max(0., result['round_seconds']-pooled),
                     plan_seconds=plan_seconds, sync_seconds=sync_seconds,
                     accepted_requests=len(batch), rejected_requests=0)
        events.append(event)
        if on_event:
            on_event(event)
    return events


def summarize(events):
    import numpy as np
    lat = [v for e in events for v in e['latency_seconds']]
    wait = [v for e in events for v in e['waiting_seconds']]
    late = [v for e in events for v in e['lateness_seconds']]
    n = len(lat)
    return dict(requests=n, events=len(events), deadline_misses=sum(e['deadline_misses'] for e in events),
                deadline_success_rate=sum(x == 0 for x in late)/n,
                joint_success_requests=sum(sum(x == 0 for x in e['lateness_seconds'])
                                           for e in events if not e['round_over_budget']),
                latency_mean_seconds=float(np.mean(lat)), latency_p95_seconds=float(np.quantile(lat, .95)),
                latency_max_seconds=max(lat), waiting_mean_seconds=float(np.mean(wait)),
                service_seconds=sum(e['service_seconds'] for e in events),
                round_seconds=sum(e['round_seconds'] for e in events),
                lateness_total_seconds=sum(late), round_over_budget_events=sum(e['round_over_budget'] for e in events),
                forced_over_budget_events=sum(e['forced_over_budget'] for e in events),
                service_over_budget_events=sum(e['service_over_budget'] for e in events),
                round_overrun_seconds=sum(e['round_overrun_seconds'] for e in events),
                within_budget_requests=sum(e['requests'] for e in events if not e['round_over_budget']),
                full_requests=sum(e['requests'] for e in events if e['policy'] == 'shard_full'),
                rows_x_rounds=sum(e['rows_x_rounds'] for e in events),
                makespan_seconds=events[-1]['model_ready_seconds'])
