"""Stress injection for the Digital Twin experiments, through the worker_factory hook of src/fu/shard_exact.py.

A StressController is installed on a ShardedUnlearner (or a fork of it): every shard retraining then builds its client workers
through controller.factory, which wraps the ordinary ClientWorker in a StressedWorker when the current condition asks for it.
The wrapper calls the real worker with the same arguments and returns its update unchanged, so the trained weights are
bit-identical to an unstressed run; only time changes. It then really sleeps, so the round wall time that the CostMeter
measures (fedavg's wall_seconds) contains the injected delay:

- slow client (factor f >= 1): the client computes f times slower. It sleeps (f - 1) x its measured compute time, and its
  reported compute_seconds becomes f x the measured value (what the client itself would measure);
- reduced bandwidth (b bits/s): the model goes down and the update comes up over the link, 2 x payload_bytes x 8 / b seconds
  per client and round. This is transfer, not compute: it is not added to compute_seconds and so lands in the round's server
  share (wall - sum of client compute), where the RPC prototype's network time lands too.

Stale state is not an environment effect but a property of the twin (its telemetry lags reality); it is applied by the
planner (src/dt/planner.py, `lag`), not here.

The condition is chosen per request: the runner sets controller.step to the 1-based index of the request before applying it,
and ENVIRONMENTS maps (step) -> Condition. Clients made slow are drawn once per environment, one per shard, with a seeded
generator, so every shard carries one slow client.
"""
import math
import time
from dataclasses import dataclass, field

import numpy as np

from src.fl.fedavg_core import ClientWorker

ENVIRONMENTS = ('none', 'slow_clients', 'low_bandwidth', 'transient_slow')
STRESS = {'onset': 6, 'transient_end': 12, 'slow_factor': 4., 'bandwidth_bps': 2e6, 'seed_slow': 301}


@dataclass(frozen=True)
class Condition:
    slow: dict = field(default_factory=dict)      # client -> compute factor (>= 1)
    bandwidth_bps: float = math.inf

    @property
    def active(self):
        return bool(self.slow) or math.isfinite(self.bandwidth_bps)

    def transfer_seconds(self, payload_bytes):
        """Seconds to send the model down and the update up once."""
        return 0. if not math.isfinite(self.bandwidth_bps) else 2*payload_bytes*8/self.bandwidth_bps


class StressedWorker:
    def __init__(self, inner, factor=1., transfer_seconds=0., sleep=time.sleep):
        if factor < 1 or transfer_seconds < 0:
            raise ValueError('factor must be >= 1 and transfer_seconds >= 0')
        self.inner, self.factor, self.transfer, self.sleep = inner, float(factor), float(transfer_seconds), sleep
        self.client_id = inner.client_id

    def train(self, global_model, config, seed):
        update, count, log = self.inner.train(global_model, config, seed)
        slow = (self.factor - 1)*log['compute_seconds']
        if slow + self.transfer > 0:
            self.sleep(slow + self.transfer)
        return update, count, {**log, 'compute_seconds': log['compute_seconds'] + slow, 'injected_seconds': slow + self.transfer}


def slow_clients(assignment, seed=STRESS['seed_slow']):
    """One client per shard, drawn with a seeded generator (the same clients for every request and training seed)."""
    rng = np.random.default_rng(seed)
    return sorted(int(rng.choice(sorted(a))) for a in assignment if a)


def environment(name, assignment, cfg=STRESS):
    """(step) -> Condition for one named environment; steps are the 1-based request indices."""
    if name not in ENVIRONMENTS:
        raise ValueError(f'unknown environment {name!r}')
    slow = {c: cfg['slow_factor'] for c in slow_clients(assignment, cfg['seed_slow'])}
    if name == 'none':
        return lambda step: Condition()
    if name == 'slow_clients':
        return lambda step: Condition(slow=slow) if step >= cfg['onset'] else Condition()
    if name == 'low_bandwidth':
        return lambda step: Condition(bandwidth_bps=cfg['bandwidth_bps']) if step >= cfg['onset'] else Condition()
    return lambda step: Condition(slow=slow) if cfg['onset'] <= step <= cfg['transient_end'] else Condition()


class StressController:
    """Builds the client workers of every retraining of `fed` under the condition of the current step."""

    def __init__(self, fed, schedule, sleep=time.sleep):
        self.fed, self.schedule, self.sleep, self.step = fed, schedule, sleep, 0
        self.workers_built = 0

    def condition(self):
        return self.schedule(self.step)

    def factory(self, client, rows):
        fed = self.fed
        tr = fed.train
        worker = ClientWorker(int(client), tr['X'][rows], tr['y'][rows], tr['row_id'][rows], fed.config.device)
        self.workers_built += 1
        cond = self.condition()
        if not cond.active:
            return worker
        return StressedWorker(worker, cond.slow.get(int(client), 1.), cond.transfer_seconds(fed.model_bytes), self.sleep)


def install(fed, schedule, sleep=time.sleep):
    """Route every later retraining of `fed` through a StressController; returns it (set .step before each request)."""
    controller = StressController(fed, schedule, sleep)
    fed.worker_factory = controller.factory
    return controller
