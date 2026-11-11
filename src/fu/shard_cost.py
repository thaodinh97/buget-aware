"""Measured cost records of sharded FedAvg training, the cost input of the Digital Twin.

One CostMeter covers one operation (the base fit, or one unlearning request). It runs every FedAvg round through the shared
core (src.fl.fedavg_core.fedavg), so the trained model is bit-identical to an unmetered run, and records:
- per client and round: samples, optimizer steps, loss, compute seconds (measured inside ClientWorker.train), and the payload
  down / up, both as raw tensor bytes and as the measured size of the serialized payload (src.fl.distributed.pack_state, the
  wire format of the Docker prototype, without HTTP headers);
- per round: wall seconds of the sequential simulation, sum and max of the client compute seconds, and the server share
  (aggregation and loop overhead = wall - sum of client compute);
- per operation: named timers (policy evaluation on the server, payload packing, checkpoint writing, the one-time warm-up of
  the numeric backend in the first operation of a process) and the resident memory of the process (start, end, peak over
  samples taken after every round).

What is simulated, not measured:
- clients run one after another in one process. `makespan_seconds_simulated` adds, per round, the slowest client and the
  server share; it is what a round would take with clients in parallel and no contention, and it excludes network transfer
  time (the Digital Twin derives that from the byte counts and its bandwidth state);
- memory is the whole simulation process (every client and every shard), not one client's footprint.
"""
import os
import time
from collections import defaultdict
from contextlib import contextmanager

from src.fl.distributed import pack_state
from src.fl.fedavg_core import fedavg

TRAIN_PHASES = ('base', 'retrain', 'fallback', 'backfill')
MAKESPAN_METHOD = ('per round: slowest client compute + server share (wall - sum of client compute); clients simulated '
                   'sequentially in one process; network transfer time excluded')


def process_rss_bytes():
    """Resident set size of this process in bytes, or None when neither psutil nor /proc is available."""
    try:
        import psutil
        return int(psutil.Process().memory_info().rss)
    except ImportError:
        pass
    try:
        with open('/proc/self/statm', encoding='ascii') as handle:
            return int(handle.read().split()[1]) * os.sysconf('SC_PAGE_SIZE')
    except (OSError, ValueError, AttributeError):
        return None


def state_bytes(model_state):
    return int(sum(v.numel() * v.element_size() for v in model_state.values()))


class CostMeter:
    def __init__(self, measure_payload=True):
        self.measure_payload = measure_payload
        self.rounds, self.clients = [], []
        self.seconds = defaultdict(float)
        self.started = time.perf_counter()
        self.rss_start = self.rss_peak = process_rss_bytes()

    @contextmanager
    def timed(self, name):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.seconds[name] += time.perf_counter() - start

    def _sample_memory(self):
        rss = process_rss_bytes()
        if rss is not None and (self.rss_peak is None or rss > self.rss_peak):
            self.rss_peak = rss
        return rss

    def _packed(self, model_state, round_index):
        if not self.measure_payload:
            return None
        with self.timed('payload_pack'):
            return len(pack_state(model_state, round_index))

    def run_round(self, model, workers, fl, seed, shard, round_index, phase):
        """One FedAvg round of `workers` from `model`; returns the aggregated model. round_index is 1-based."""
        captured = {}

        def observer(_round, before, _after, updates, _counts, _log):
            captured.update(before=before, updates=updates)

        result, logs = fedavg(model, workers, fl, 1, seed, observer=observer if self.measure_payload else None)
        log = logs[0]
        raw = log['payload_bytes_up'] // len(workers)
        down = self._packed(captured['before'], round_index) if self.measure_payload else None
        updates = captured.get('updates') or [None] * len(workers)
        for client, samples, update in zip(log['clients'], log['counts'], updates):
            self.clients.append({'shard': shard, 'round': round_index, 'phase': phase, 'client_id': int(client['client_id']),
                                 'samples': int(samples), 'optimizer_steps': int(client['steps']), 'loss': float(client['loss']),
                                 'compute_seconds': float(client['compute_seconds']), 'bytes_down_raw': raw, 'bytes_up_raw': raw,
                                 'bytes_down_packed': down,
                                 'bytes_up_packed': self._packed(update, round_index) if update is not None else None})
        self._close_round(shard, round_index, phase, len(workers), float(log['wall_seconds']))
        return result

    def record_local(self, shard, phase, records):
        """Client-local work outside FedAvg aggregation (DynamicFU contribution probes): one parallel batch of clients.
        records: [(client_id, samples, train_log, base_state, update_state)]."""
        if not records:
            return
        for client_id, samples, log, base, update in records:
            self.clients.append({'shard': shard, 'round': 0, 'phase': phase, 'client_id': int(client_id), 'samples': int(samples),
                                 'optimizer_steps': int(log['steps']), 'loss': float(log['loss']),
                                 'compute_seconds': float(log['compute_seconds']), 'bytes_down_raw': state_bytes(base),
                                 'bytes_up_raw': state_bytes(update), 'bytes_down_packed': self._packed(base, 0),
                                 'bytes_up_packed': self._packed(update, 0)})
        self._close_round(shard, 0, phase, len(records), None)

    def _close_round(self, shard, round_index, phase, count, wall):
        rows = self.clients[-count:]
        compute = [r['compute_seconds'] for r in rows]
        total = lambda key: None if any(r[key] is None for r in rows) else int(sum(r[key] for r in rows))
        wall = sum(compute) if wall is None else wall
        self.rounds.append({'shard': shard, 'round': round_index, 'phase': phase, 'clients': count,
                            'samples': int(sum(r['samples'] for r in rows)), 'wall_seconds': wall,
                            'compute_seconds_sum': float(sum(compute)), 'compute_seconds_max': float(max(compute)),
                            'server_seconds': max(0., wall - sum(compute)),
                            'bytes_down_raw': total('bytes_down_raw'), 'bytes_up_raw': total('bytes_up_raw'),
                            'bytes_down_packed': total('bytes_down_packed'), 'bytes_up_packed': total('bytes_up_packed'),
                            'rss_bytes': self._sample_memory()})

    def train_totals(self):
        """Counters of the training rounds so far; train_bytes_raw equals the byte formula of the v2 runners."""
        train = [r for r in self.rounds if r['phase'] in TRAIN_PHASES]
        return {'rounds': len(train), 'rows_x_rounds': int(sum(r['samples'] for r in train)),
                'train_bytes_raw': int(sum(r['bytes_down_raw'] + r['bytes_up_raw'] for r in train))}

    def summary(self):
        """Totals of the operation. Training counters (rounds, rows_x_rounds, clients_selected) cover TRAIN_PHASES only;
        byte and time totals cover every recorded round, contribution probes included."""
        train = [r for r in self.rounds if r['phase'] in TRAIN_PHASES]
        packed = lambda key: None if any(r[key] is None for r in self.rounds) else int(sum(r[key] for r in self.rounds))
        per_shard = defaultdict(float)
        for r in self.rounds:
            per_shard[r['shard']] += r['compute_seconds_max'] + r['server_seconds']
        policy = self.seconds.get('policy_eval', 0.)
        rss_end = self._sample_memory()
        out = {**self.train_totals(), 'client_rounds': int(sum(r['clients'] for r in train)),
               'clients_selected': sorted({c['client_id'] for c in self.clients if c['phase'] in TRAIN_PHASES}),
               'shards_trained': sorted({r['shard'] for r in train}),
               'bytes_down_raw': int(sum(r['bytes_down_raw'] for r in self.rounds)),
               'bytes_up_raw': int(sum(r['bytes_up_raw'] for r in self.rounds)),
               'bytes_down_packed': packed('bytes_down_packed'), 'bytes_up_packed': packed('bytes_up_packed'),
               'compute_seconds_sum': float(sum(r['compute_seconds_sum'] for r in self.rounds)),
               'server_aggregate_seconds': float(sum(r['server_seconds'] for r in self.rounds)),
               'policy_eval_seconds': float(policy),
               'payload_pack_seconds': float(self.seconds.get('payload_pack', 0.)),
               'backend_warmup_seconds': float(self.seconds.get('backend_warmup', 0.)),
               'checkpoint_seconds': float(self.seconds.get('checkpoint', 0.)),
               'makespan_seconds_simulated': float(sum(per_shard.values()) + policy),
               'makespan_parallel_shards_seconds_simulated': float(max(per_shard.values(), default=0.) + policy),
               'makespan_method': MAKESPAN_METHOD,
               'wall_seconds_total': time.perf_counter() - self.started,
               'rss_bytes_start': self.rss_start, 'rss_bytes_end': rss_end, 'rss_bytes_peak_sampled': self.rss_peak,
               'memory_scope': 'whole simulation process (all clients and shards), sampled after every round',
               'payload_format': 'src.fl.distributed.pack_state (compressed NPZ), HTTP headers excluded'
                                 if self.measure_payload else 'not measured'}
        return out
