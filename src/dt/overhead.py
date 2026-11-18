"""What the Digital Twin itself costs: the time of its own computations and the telemetry it needs.

The twin's work around one request (src/dt/planner.Twin):
  state  read its view of the touched shard, samples per client after the deletion, from the live system
  plan   forecast both options, the margin factors, the decision
  sync   fold the cost records of the finished operation into the predictor and the error histories
dt_overhead_seconds of the implementation guide (section 7.2) = state + plan + sync.

Telemetry is what an operation must report for the sync. Two sizes are given:
  records  the per-round and per-client-round cost records the twin consumes today (the fields below), as JSON, and the
           same text compressed with zlib;
  summary  the sufficient statistics of one operation: per client the total compute seconds and samples, plus the shard, the
           number of rounds, the round wall time (in total and up to the checkpoint) and the server seconds. summary_rates()
           recovers exactly the profile that planner.rates() fits from the records, so the twin could be fed with the summary
           alone; the tests check this.
Both are compared with the model payload of the same operation (2 x rounds x clients x model bytes, raw tensors).

Nothing here trains or measures a network: a byte count is a size, not a transfer time. Times are of the Python code of this
repository on the machine that runs it.
"""
import json
import time
import zlib
from statistics import median

from src.dt import planner

ROUND_FIELDS = ('shard', 'round', 'samples', 'wall_seconds', 'server_seconds')
CLIENT_FIELDS = ('round', 'client_id', 'samples', 'compute_seconds')


def operation_summary(rounds, clients, fixed_rounds=planner.R_FIXED):
    """Sufficient statistics of one operation for every predictor and margin of src/dt/planner.py."""
    per_client = {}
    for c in clients:
        seconds, samples = per_client.get(c['client_id'], (0., 0))
        per_client[c['client_id']] = (seconds + c['compute_seconds'], samples + c['samples'])
    return {'shard': rounds[0].get('shard'), 'rounds': len(rounds),
            'wall_seconds': sum(r['wall_seconds'] for r in rounds),
            'wall_seconds_until_checkpoint': sum(r['wall_seconds'] for r in rounds if r['round'] <= fixed_rounds),
            'server_seconds': sum(r['server_seconds'] for r in rounds),
            'clients': {str(c): [seconds, samples] for c, (seconds, samples) in per_client.items()}}


def summary_rates(summary):
    """The profile planner.rates() fits from the full records, from the summary alone."""
    return {'client': {c: seconds/samples for c, (seconds, samples) in summary['clients'].items()},
            'server': summary['server_seconds']/summary['rounds']}


def telemetry_bytes(rounds, clients, fixed_rounds=planner.R_FIXED):
    records = json.dumps({'rounds': [{k: r[k] for k in ROUND_FIELDS if k in r} for r in rounds],
                          'clients': [{k: c[k] for k in CLIENT_FIELDS} for c in clients]}, separators=(',', ':')).encode()
    summary = json.dumps(operation_summary(rounds, clients, fixed_rounds), separators=(',', ':')).encode()
    return {'records_json': len(records), 'records_zlib': len(zlib.compress(records, 9)), 'summary_json': len(summary)}


def time_twin(requests, cfg, fit_profile, fit_history, fit_records=None, budget=float('inf'), repeats=5):
    """Seconds the twin spends per request on plan and sync, replaying a measured trajectory `repeats` times (median per
    request). Every request is treated as accepted with shard_full, so the sync always folds 20 rounds of records, the
    largest case. Returns [{'step', 'plan_seconds', 'sync_seconds'}]. The state part needs the live system and is not here."""
    plan = [[] for _ in requests]
    sync = [[] for _ in requests]
    for _ in range(repeats):
        twin = planner.Twin(cfg, fit_profile, fit_history, fit_records)
        for k, req in enumerate(requests):
            counts = planner.twin_counts(requests, k, cfg.lag)
            started = time.perf_counter()
            f_full, f_fixed = twin.forecasts(counts, req['shard'])
            twin.factors()
            twin.start(f_full, f_fixed, budget)
            plan[k].append(time.perf_counter() - started)
            spent = planner.cost(req, cfg.full_rounds)
            records = planner.observed_records(req, cfg.full_rounds)
            started = time.perf_counter()
            twin.record(k, records, spent/f_full, planner.extrapolation_ratio(planner.cost(req, cfg.fixed_rounds), spent,
                                                                              cfg.full_rounds, cfg.fixed_rounds))
            twin.advance(k + 1 + cfg.lag)              # deliver this observation now, whatever the lag, to time the sync
            sync[k].append(time.perf_counter() - started)
    return [{'step': req['step'], 'plan_seconds': median(plan[k]), 'sync_seconds': median(sync[k])}
            for k, req in enumerate(requests)]


class Stopwatch:
    """Named single-shot timers for the live loop: `with watch('plan'): ...`; watch.seconds['plan']."""

    def __init__(self):
        self.seconds = {}
        self._name = self._started = None

    def __call__(self, name):
        self._name = name
        return self

    def __enter__(self):
        self._started = time.perf_counter()

    def __exit__(self, *exc):
        self.seconds[self._name] = self.seconds.get(self._name, 0.) + time.perf_counter() - self._started
        return False
