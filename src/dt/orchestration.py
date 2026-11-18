"""Controllers of the C1/C2 orchestration protocol (docs/dt_orchestration_plan.md).

Every arm uses the real-time contract of src/dt/realtime.py and src/dt/predictive.py: open-loop arrivals on the
clock, one worker, only ARRIVED requests visible, every request eventually executed and costed (never rejected),
feedback only from executed work, and the same event record. realtime.py and predictive.py are hash-locked by the
earlier DT protocols, so they are not edited; the proposed controller is re-implemented here with switches, and a test
checks that with the default switches it takes exactly the decisions of predictive.run.

Proposed controller (`queue_run`) and its C2 ablations:
- order='enumerate' (default) plans the order of all pending shards; order='edf' fixes earliest-deadline order
  (after the same never-idle rule) and only plans full/fixed per shard.
- overhead=True (default) plans with service time (rounds + learned per-client-round overhead); overhead=False plans
  with round time only.
- pooling=True (default) waits for more arrivals when even shard_fixed is predicted over the pooled budget;
  pooling=False never waits.

C1 baselines (`baseline_run`), after strategies in the literature:
- 'immediate': whenever the worker is free, serve the queue holding the oldest request, shard_full. No forecast in the
  decision (retrain per request; requests that arrive during a retrain are batched by necessity).
- 'window': at fixed boundaries k*W, serve the requests that arrived by the boundary, shard_full (SISA-style
  buffering of requests). No forecast in the decision.
- 'lazy': postpone each shard's queue until the latest start of shard_full before its oldest deadline, then serve it
  with everything queued, shard_full (ERASER-style postponement to batch). Earliest deadline first among due queues.
- 'edf': whenever the worker is free, serve the earliest-deadline queue; shard_full if its forecast fits the pooled
  budget and the deadline, else shard_fixed; never waits.
Baselines receive forecasts from the same twin as the proposed controller (for 'lazy' and 'edf' decisions, and for
the logged forced_over_budget flag); the twin learns from their executed work in the same way.
"""
import itertools
import math
import time

POLICIES = ('shard_full', 'shard_fixed')
BASELINES = ('immediate', 'window', 'lazy', 'edf')
ORDERS = ('enumerate', 'edf')


def _validate(requests, budget, deadline):
    if not requests or budget <= 0 or deadline <= 0:
        raise ValueError('nonempty workload and positive budget/deadline required')
    if any(r['time'] < 0 for r in requests) or any(a['time'] > b['time'] for a, b in zip(requests, requests[1:])):
        raise ValueError('arrival times must be nonnegative and sorted')


def _forecasts(twin, shard, batch, full_rounds, fixed_rounds):
    predictions = {policy: twin.forecast(shard, batch, rounds)
                   for policy, rounds in (('shard_full', full_rounds), ('shard_fixed', fixed_rounds))}
    if not all(math.isfinite(p[k]) and p[k] > 0 for p in predictions.values()
               for k in ('mean_round', 'upper_round', 'service_upper')):
        raise ValueError('invalid forecast')
    return predictions


def _event(events, result, shard, batch, policy, prediction, started, ready, budget, deadline,
           plan_seconds, sync_seconds, rollout):
    """The event record of predictive.run, field for field."""
    latency = [ready-r['time'] for r in batch]
    pooled = budget*len(batch)
    event = {k: v for k, v in result.items() if k not in ('round_costs', 'client_costs')}
    event.update(event=len(events)+1, shard=shard, steps=[r['step'] for r in batch],
                 arrivals=[r['time'] for r in batch], requests=len(batch), policy=policy,
                 started_seconds=started, model_ready_seconds=ready, service_seconds=ready-started,
                 waiting_seconds=[started-r['time'] for r in batch], latency_seconds=latency,
                 deadline_seconds=deadline, lateness_seconds=[max(0., t-deadline) for t in latency],
                 deadline_misses=sum(t > deadline for t in latency), pooled_budget_seconds=pooled,
                 forecast_round_seconds=prediction['mean_round'], margin_factor=prediction['margin'],
                 forecast_service_seconds=prediction['service_upper'], forecast_details=prediction,
                 rollout=rollout, forced_over_budget=prediction['upper_round'] > pooled,
                 round_over_budget=result['round_seconds'] > pooled, service_over_budget=ready-started > pooled,
                 round_overrun_seconds=max(0., result['round_seconds']-pooled),
                 plan_seconds=plan_seconds, sync_seconds=sync_seconds, accepted_requests=len(batch),
                 rejected_requests=0)
    return event


def choose(jobs, now, budget, deadline, order='enumerate', overhead=True):
    """predictive.choose with the C2 switches; the defaults reproduce it (same orders, ties and score)."""
    if not jobs or len(jobs) > 4:
        raise ValueError('planner supports one to four pending shards')
    if order not in ORDERS:
        raise ValueError(f'order must be one of {ORDERS}')
    key = 'service_upper' if overhead else 'upper_round'
    affordable_first = {s for s, (batch, predictions) in jobs.items()
                        if predictions['shard_fixed']['upper_round'] <= budget*len(batch)}
    if order == 'enumerate':
        orders = [o for o in itertools.permutations(sorted(jobs)) if not affordable_first or o[0] in affordable_first]
    else:
        edf = sorted(jobs, key=lambda s: (jobs[s][0][0]['time'], s))
        if affordable_first:
            first = next(s for s in edf if s in affordable_first)
            edf = [first] + [s for s in edf if s != first]
        orders = [tuple(edf)]
    best, evaluated = None, 0
    for sequence in orders:
        for policies in itertools.product(POLICIES, repeat=len(sequence)):
            cursor, success, over, quality, late, latency = now, 0, 0, 0, 0., 0.
            for shard, policy in zip(sequence, policies):
                batch, forecasts = jobs[shard]
                prediction = forecasts[policy]
                cursor += prediction[key]
                affordable = prediction['upper_round'] <= budget*len(batch)
                over += 0 if affordable else len(batch)
                for request in batch:
                    duration = cursor-request['time']
                    ontime = duration <= deadline
                    success += int(ontime and affordable)
                    quality += int(ontime and affordable and policy == 'shard_full')
                    late += max(0., duration-deadline)
                    latency += duration
            score = (success, -over, quality, -late, -latency)
            evaluated += 1
            if best is None or score > best['score']:
                best = dict(score=score, shard=sequence[0], policy=policies[0], order=list(sequence),
                            policies=list(policies), predicted_joint_success=success)
    best['plans_evaluated'] = evaluated
    return best


def queue_run(requests, twin, execute, budget, deadline, full_rounds=20, fixed_rounds=10, order='enumerate',
              overhead=True, pooling=True, clock=time.perf_counter, sleep=time.sleep, on_event=None):
    """predictive.run with the C2 switches; with the defaults it takes the same decisions."""
    _validate(requests, budget, deadline)
    key = 'service_upper' if overhead else 'upper_round'
    origin, i, queues, events = clock(), 0, {}, []
    planning_total = 0.
    while i < len(requests) or any(queues.values()):
        now = clock()-origin
        while i < len(requests) and requests[i]['time'] <= now:
            queues.setdefault(requests[i]['shard'], []).append(requests[i])
            i += 1
        if not any(queues.values()):
            sleep(max(0., requests[i]['time']-(clock()-origin)))
            continue
        plan_start = clock()
        jobs = {shard: (batch, _forecasts(twin, shard, batch, full_rounds, fixed_rounds))
                for shard, batch in queues.items() if batch}
        choice = choose(jobs, clock()-origin, budget, deadline, order, overhead)
        planning_total += clock()-plan_start
        shard, policy = choice['shard'], choice['policy']
        batch, forecasts = jobs[shard]
        prediction = forecasts[policy]
        current = clock()-origin
        fixed = forecasts['shard_fixed']
        cutoff = batch[0]['time']+deadline-fixed[key]
        if pooling and fixed['upper_round'] > len(batch)*budget and cutoff-current > .01:
            wake = min(cutoff, current+max(.01, .25*fixed[key]))
            if i < len(requests):
                wake = min(wake, requests[i]['time'])
            sleep(max(0., wake-(clock()-origin)))
            continue
        queues[shard] = []
        started = clock()-origin
        result = execute(batch, policy, started)
        ready = clock()-origin
        observed = clock()
        twin.observe(shard, result, prediction, ready-started)
        sync_seconds = clock()-observed
        events.append(_event(events, result, shard, batch, policy, prediction, started, ready, budget, deadline,
                             planning_total, sync_seconds, choice))
        planning_total = 0.
        if on_event:
            on_event(events[-1])
    return events


def _oldest(queues):
    return min(queues, key=lambda s: (queues[s][0]['time'], s))


def baseline_run(requests, twin, execute, budget, deadline, mode, window=None, full_rounds=20, fixed_rounds=10,
                 clock=time.perf_counter, sleep=time.sleep, on_event=None):
    """One of BASELINES on the same loop; see the module docstring."""
    _validate(requests, budget, deadline)
    if mode not in BASELINES:
        raise ValueError(f'mode must be one of {BASELINES}')
    if mode == 'window' and not (window and window > 0):
        raise ValueError('window mode needs a positive window length')
    origin, i, queues, events = clock(), 0, {}, []
    planning_total = 0.
    while i < len(requests) or any(queues.values()):
        now = clock()-origin
        while i < len(requests) and requests[i]['time'] <= now:
            queues.setdefault(requests[i]['shard'], []).append(requests[i])
            i += 1
        if not any(queues.values()):
            sleep(max(0., requests[i]['time']-(clock()-origin)))
            continue
        plan_start = clock()
        pending = {s: b for s, b in queues.items() if b}
        now = clock()-origin
        shard, batch, policy, wake = None, None, 'shard_full', None
        if mode == 'window':
            boundary = math.floor(now/window + 1e-9)*window
            eligible = {s: [r for r in b if r['time'] <= boundary] for s, b in pending.items()}
            eligible = {s: b for s, b in eligible.items() if b}
            if eligible:
                shard = _oldest(eligible)
                batch = eligible[shard]
            else:
                wake = boundary+window
        elif mode in ('immediate', 'edf'):
            shard = _oldest(pending)
            batch = pending[shard]
        else:  # lazy
            latest = {s: b[0]['time']+deadline-_forecasts(twin, s, b, full_rounds, fixed_rounds)['shard_full']['service_upper']
                      for s, b in pending.items()}
            due = {s: b for s, b in pending.items() if now >= latest[s]-1e-9}
            if due:
                shard = _oldest(due)
                batch = due[shard]
            else:
                wake = min(latest.values())
        forecasts = None
        if shard is not None:
            forecasts = _forecasts(twin, shard, batch, full_rounds, fixed_rounds)
            if mode == 'edf':
                full = forecasts['shard_full']
                fits = full['upper_round'] <= budget*len(batch) and now+full['service_upper'] <= batch[0]['time']+deadline
                policy = 'shard_full' if fits else 'shard_fixed'
        planning_total += clock()-plan_start
        if shard is None:
            if i < len(requests):
                wake = min(wake, requests[i]['time'])
            sleep(max(0., wake-(clock()-origin)))
            continue
        served = {r['step'] for r in batch}
        queues[shard] = [r for r in queues[shard] if r['step'] not in served]
        prediction = forecasts[policy]
        started = clock()-origin
        result = execute(batch, policy, started)
        ready = clock()-origin
        observed = clock()
        twin.observe(shard, result, prediction, ready-started)
        sync_seconds = clock()-observed
        events.append(_event(events, result, shard, batch, policy, prediction, started, ready, budget, deadline,
                             planning_total, sync_seconds, dict(mode=mode)))
        planning_total = 0.
        if on_event:
            on_event(events[-1])
    return events
