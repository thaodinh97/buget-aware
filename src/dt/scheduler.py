"""Deadline-aware batching of deletion requests: direction 3 of docs/dt_extensions.md.

Time. Requests arrive at the simulated times of the v3 stream (scripts/shard/stream_v3.py, Poisson, rate 1). Processing is
instantaneous in simulated time, as in stream_v3.cost_plan: its compute seconds count against the budget, not against the
clock. A request is served by the processing event that retrains its shard after its deletion; latency = service time -
arrival time; the deletion deadline is latency <= D.

Budget. Every request brings B seconds of FedAvg round wall time (the per-request budget of directions 1-2). An event that
serves n requests may spend n x B: the budget is pooled over the requests the event serves. A request over budget on its own
therefore waits in the queue of its shard instead of being rejected, and is served with the next requests of that shard.

Modes (when to process, with which policy):
  immediate  every request at its arrival (stream_v3 W = 0), shard_full
  window     everything queued, at the end of each window [mW, (m+1)W) (the v3 policy batch_full_W), shard_full
  lazy       per shard: its queue when the oldest request of the queue reaches its deadline, shard_full
  dt         per shard, after every arrival: the shard's queue now with shard_full if the twin's forecast of one retrain,
             times its margin factor, fits the pooled budget of the queue; otherwise wait. When the oldest request of the
             queue reaches its deadline the queue is processed in any case: shard_full if it fits, else shard_fixed if it
             fits, else shard_fixed anyway (the deadline is hard; the event is marked forced_over_budget)
Simultaneous events: due events are processed before an arrival at the same time (stream_v3 puts a request that arrives at
mW into the window that starts there); among due shards, the one with the earliest due time, then the lowest index.

The scheduler does not train: execute(batch, policy) applies the deletions of `batch` (requests in arrival order) and returns
a dict with at least 'seconds' (measured round wall time). In dt mode, `twin` forecasts and learns (DtTwin below).
"""
import math

from src.dt import planner

MODES = ('immediate', 'window', 'lazy', 'dt')


class DtTwin:
    """The twin of the dt mode: Adaptive profile (EWMA .3) and the error margin of src/dt/planner.py.
    counts(shard, queued) -> {client: samples after the queued deletions}, from the live state of the unlearner."""

    def __init__(self, profile, history, counts, margin_q=.9, window=10):
        self.profile, self.counts = profile, counts
        self.margin = planner.Margin(margin_q, history, window)

    def forecast(self, shard, queued, rounds):
        return planner.forecast(self.profile, self.counts(shard, queued), rounds)

    def factor(self):
        return self.margin.factor()

    def observe(self, result, predicted):
        if result.get('round_costs'):
            self.profile = planner.ewma(self.profile, planner.rates(result['round_costs'], result['client_costs']))
        if predicted:
            self.margin.observe(result['seconds']/predicted)


def run(requests, mode, deadline, budget, execute, window=None, twin=None, rounds=(planner.R_FULL, planner.R_FIXED)):
    """Schedule `requests` ({'step', 'time', 'shard', 'rows'}, sorted by time). rounds: (R_base, shard_fixed_rounds) of
    the train config, for the twin's forecasts. Returns the list of processing events."""
    if mode not in MODES:
        raise ValueError(f'mode must be one of {MODES}')
    if mode == 'window' and not (window and window > 0):
        raise ValueError('window mode needs window > 0')
    if mode == 'dt' and twin is None:
        raise ValueError('dt mode needs a twin')
    if mode in ('lazy', 'dt') and not deadline > 0:
        raise ValueError('lazy and dt modes need a deadline > 0')
    times = [r['time'] for r in requests]
    if times != sorted(times):
        raise ValueError('requests must be sorted by arrival time')
    queues, events, i = {}, [], 0

    def due():
        """(time, shards) of the next processing event that is not an arrival, or (inf, [])."""
        live = {j: q for j, q in queues.items() if q}
        if not live:
            return math.inf, []
        if mode == 'window':
            first = min(q[0]['time'] for q in live.values())
            return (math.floor(first/window) + 1)*window, sorted(live)
        if mode in ('lazy', 'dt'):
            j = min(live, key=lambda s: (live[s][0]['time'], s))
            return live[j][0]['time'] + deadline, [j]
        return math.inf, []

    def process(t, shards, policy, forced=False, predicted=None, pooled_forecast=None):
        batch = sorted((r for j in shards for r in queues[j]), key=lambda r: (r['time'], r['step']))
        for j in shards:
            queues[j] = []
        result = execute(batch, policy)
        pooled = budget*len(batch)
        latencies = [t - r['time'] for r in batch]
        event = {'time': t, 'mode': mode, 'policy': policy, 'shards': sorted(shards), 'steps': [r['step'] for r in batch],
                 'requests': len(batch), 'rows': int(sum(r['rows'] for r in batch)), 'latencies': latencies,
                 'exposure': float(sum(r['rows']*lat for r, lat in zip(batch, latencies))),
                 'pooled_budget': pooled, 'forecast_seconds': predicted, 'forecast_with_margin': pooled_forecast,
                 'forced_over_budget': forced,
                 **{k: v for k, v in result.items() if k not in ('round_costs', 'client_costs')}}
        event['overrun_seconds'] = max(0., result['seconds'] - pooled)
        event['over_budget'] = result['seconds'] > pooled
        event['deadline_missed'] = any(lat > deadline + 1e-9 for lat in latencies) if deadline else False
        events.append(event)
        if twin is not None:
            twin.observe(result, predicted)

    def dt_choice(j, at_deadline):
        q = queues[j]
        full = twin.forecast(j, q, rounds[0])
        fixed = twin.forecast(j, q, rounds[1])
        m, pooled = twin.factor(), budget*len(q)
        if full*m <= pooled:
            return 'shard_full', full, full*m, False
        if not at_deadline:
            return None, full, full*m, False
        return 'shard_fixed', fixed, fixed*m, fixed*m > pooled

    while i < len(requests) or any(queues.values()):
        arrival = requests[i]['time'] if i < len(requests) else math.inf
        t_due, shards = due()
        if t_due <= arrival:
            if mode == 'dt':
                policy, predicted, with_margin, forced = dt_choice(shards[0], True)
                process(t_due, shards, policy, forced, predicted, with_margin)
            else:
                process(t_due, shards, 'shard_full')
            continue
        r = requests[i]; i += 1
        queues.setdefault(r['shard'], []).append(r)
        if mode == 'immediate':
            process(r['time'], [r['shard']], 'shard_full')
        elif mode == 'dt':
            policy, predicted, with_margin, _ = dt_choice(r['shard'], False)
            if policy is not None:
                process(r['time'], [r['shard']], policy, False, predicted, with_margin)
    return events


def summarize(events, deadline):
    lat = [x for e in events for x in e['latencies']]
    served = sum(e['requests'] for e in events)
    full = sum(e['requests'] for e in events if e['policy'] == 'shard_full')
    return {'events': len(events), 'requests_served': served, 'retrains': sum(len(e['shards']) for e in events),
            'seconds_total': float(sum(e['seconds'] for e in events)),
            'rows_x_rounds': int(sum(e.get('rows_x_rounds', 0) for e in events)),
            'latency_mean': float(sum(lat)/len(lat)) if lat else float('nan'), 'latency_max': float(max(lat, default=float('nan'))),
            'exposure_row_time': float(sum(e['exposure'] for e in events)),
            'deadline_misses': int(sum(x > deadline + 1e-9 for x in lat)) if deadline else 0,
            'over_budget_events': int(sum(e['over_budget'] for e in events)),
            'forced_over_budget_events': int(sum(e['forced_over_budget'] for e in events)),
            'overrun_seconds': float(sum(e['overrun_seconds'] for e in events)),
            'served_full_share': full/served if served else float('nan')}
