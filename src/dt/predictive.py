"""Experimental service-aware twin and planning over the ARRIVED shard queues.

The frozen realtime.py is left unchanged. This controller plans full/fixed and
order jointly over at most four pending shards; it never sees future arrivals.
This is a bounded look-ahead controller, not an optimality/SLA guarantee.
"""
import itertools
import math
import time
from collections import deque

import numpy as np

from src.dt import planner


class ServiceTwin:
    """Predict round cost, learn model disagreement and measured service overhead.

    hybrid=True combines structural EWMA(.3), structural EWMA(1), last-shard and
    row EWMA(.3). Weights depend only on previous pre-update log forecast errors.
    hybrid=False uses row EWMA alone: the ablation gets the SAME controller,
    overhead model and margin. Neither variant observes unexecuted alternatives.
    """
    def __init__(self, fit, counts, hybrid=True, q=.9, window=10, loss_alpha=.3, eta=3., floor=.1):
        self.counts, self.q, self.window = counts, q, window
        self.loss_alpha, self.eta, self.floor = loss_alpha, eta, floor
        rs, cs = fit['round_costs'], fit['client_costs']
        specs = [('structural_03', 'adaptive', .3), ('structural_1', 'adaptive', 1.),
                 ('last_shard', 'last_shard', .3), ('rows_ewma', 'rows_ewma', .3)]
        if not hybrid:
            specs = specs[-1:]
        self.names = [x[0] for x in specs]
        self.experts = [planner.make_predictor(name, planner.rates(rs, cs), rs, alpha) for _, name, alpha in specs]
        self.losses = np.zeros(len(specs))
        initial = []
        for shard in sorted({r['shard'] for r in rs}):
            selected = [r for r in rs if r['shard'] == shard]
            counts0 = {r['client_id']: r['samples'] for r in cs if r['shard'] == shard and r['round'] == 1}
            predicted = np.mean([e.forecast(shard, counts0, len(selected)) for e in self.experts])
            initial.append(sum(r['wall_seconds'] for r in selected)/predicted)
        self.ratios = deque(initial, maxlen=window)
        # Packing is measured during fit. Validation/fit setup is NOT treated as
        # per-operation overhead. Later updates include the entire actual service.
        pack = fit.get('cost', {}).get('payload_pack_seconds', 0.)
        self.overheads = deque([max(0., pack)/max(1, len(cs))], maxlen=window)

    def weights(self):
        scores = np.exp(-self.eta*(self.losses - self.losses.min()))
        return (1-self.floor)*scores/scores.sum() + self.floor/len(scores)

    def forecast(self, shard, batch, rounds):
        counts = self.counts(shard, batch)
        expert = np.array([e.forecast(shard, counts, rounds) for e in self.experts])
        weights = self.weights()
        mean = float(weights @ expert)
        margin = max(1., float(np.quantile(list(self.ratios), self.q)))
        overhead = .05 + rounds*len(counts)*float(np.quantile(list(self.overheads), self.q))
        return dict(rounds=rounds, mean_round=mean, upper_round=mean*margin,
                    service_upper=mean*margin+overhead, overhead_seconds=overhead,
                    margin=margin, experts=expert.tolist(), weights=weights.tolist())

    def observe(self, shard, result, predicted, service_seconds):
        actual = result['round_seconds']
        errors = np.abs(np.log(np.maximum(actual, 1e-12)/np.maximum(predicted['experts'], 1e-12)))
        self.losses = (1-self.loss_alpha)*self.losses + self.loss_alpha*errors
        self.ratios.append(actual/predicted['mean_round'])
        self.overheads.append(max(0., service_seconds-actual)/max(1, len(result['client_costs'])))
        for expert in self.experts:
            expert.update(shard, result['round_costs'], result['client_costs'])


def choose(jobs, now, budget, deadline):
    """Enumerate <=4! * 2^4 plans. Choose only the FIRST action, then replan.

    Score: joint-success requests; fewer budget-violating requests; more on-time
    full requests; smaller total lateness; smaller sum of completion latencies.
    All scores use forecasts of the CURRENT arrived jobs, not true future costs.
    """
    if not jobs or len(jobs) > 4:
        raise ValueError('planner supports one to four pending shards')
    best = None
    evaluated = 0
    affordable_first = {s for s, (batch, predictions) in jobs.items()
                        if predictions['shard_fixed']['upper_round'] <= budget*len(batch)}
    for order in itertools.permutations(sorted(jobs)):
        # Never idle for budget pooling while another arrived shard can run.
        # Otherwise a wait inserted by run() would invalidate this rollout.
        if affordable_first and order[0] not in affordable_first:
            continue
        for policies in itertools.product(('shard_full', 'shard_fixed'), repeat=len(order)):
            cursor, success, over, quality, late, latency = now, 0, 0, 0, 0., 0.
            for shard, policy in zip(order, policies):
                batch, forecasts = jobs[shard]
                prediction = forecasts[policy]
                cursor += prediction['service_upper']
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
                best = dict(score=score, shard=order[0], policy=policies[0], order=list(order),
                            policies=list(policies), predicted_joint_success=success)
    best['plans_evaluated'] = evaluated
    return best


def run(requests, twin, execute, budget, deadline, full_rounds=20, fixed_rounds=10,
        clock=time.perf_counter, sleep=time.sleep, on_event=None):
    if not requests or budget <= 0 or deadline <= 0:
        raise ValueError('nonempty workload and positive budget/deadline required')
    if any(r['time'] < 0 for r in requests) or any(a['time'] > b['time'] for a, b in zip(requests, requests[1:])):
        raise ValueError('arrival times must be nonnegative and sorted')
    origin, i, queues, events = clock(), 0, {}, []
    planning_total = 0.
    while i < len(requests) or any(queues.values()):
        now = clock()-origin
        while i < len(requests) and requests[i]['time'] <= now:
            request = requests[i]
            queues.setdefault(request['shard'], []).append(request)
            i += 1
        if not any(queues.values()):
            sleep(max(0., requests[i]['time']-(clock()-origin)))
            continue
        plan_start = clock()
        jobs = {}
        for shard, batch in queues.items():
            if not batch:
                continue
            predictions = {policy: twin.forecast(shard, batch, rounds) for policy, rounds in
                           [('shard_full', full_rounds), ('shard_fixed', fixed_rounds)]}
            if not all(math.isfinite(p[k]) and p[k] > 0 for p in predictions.values()
                       for k in ('mean_round', 'upper_round', 'service_upper')):
                raise ValueError('invalid forecast')
            jobs[shard] = (batch, predictions)
        choice = choose(jobs, clock()-origin, budget, deadline)
        planning_total += clock()-plan_start
        shard, policy = choice['shard'], choice['policy']
        batch, forecasts = jobs[shard]
        prediction = forecasts[policy]
        current = clock()-origin
        # Pool only when even fixed is predicted unaffordable. Never wait for a
        # possible future full merely because fixed is the only affordable option.
        fixed = forecasts['shard_fixed']
        cutoff = batch[0]['time']+deadline-fixed['service_upper']
        if fixed['upper_round'] > len(batch)*budget and cutoff-current > .01:
            wake = min(cutoff, current+max(.01, .25*fixed['service_upper']))
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
                     rollout=choice, forced_over_budget=prediction['upper_round'] > pooled,
                     round_over_budget=result['round_seconds'] > pooled, service_over_budget=ready-started > pooled,
                     round_overrun_seconds=max(0., result['round_seconds']-pooled),
                     plan_seconds=planning_total, sync_seconds=sync_seconds, accepted_requests=len(batch),
                     rejected_requests=0)
        planning_total = 0.
        events.append(event)
        if on_event:
            on_event(event)
    return events
