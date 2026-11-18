import itertools
import unittest

import numpy as np

from src.dt import orchestration, predictive, realtime
from test_dt_realtime import Clock, request


def prediction(round_seconds, overhead=0.):
    return dict(mean_round=round_seconds, upper_round=round_seconds, service_upper=round_seconds+overhead,
                margin=1., experts=[round_seconds], weights=[1.])


class ShardTwin:
    """Deterministic twin: per-shard seconds for 20 rounds, a fixed overhead; records what it observed."""
    def __init__(self, full=None, overhead=0.):
        self.full, self.overhead, self.seen = full or {}, overhead, []
    def forecast(self, shard, batch, rounds):
        return prediction(self.full.get(shard, 2.)*rounds/20, self.overhead)
    def observe(self, shard, result, forecast, seconds):
        self.seen.append((shard, round(seconds, 9)))


def executor(clock, seconds=None, calls=None):
    def run(batch, policy, started):
        if calls is not None:
            calls.append(([r['step'] for r in batch], policy, started))
        actual = (seconds or {}).get(batch[0]['shard'], 2.)*(1 if policy == 'shard_full' else .5)
        clock.sleep(actual)
        return dict(round_seconds=actual, rows_x_rounds=10*len(batch))
    return run


def stream(seed, n=10, shards=3, gap=.8):
    rng = np.random.default_rng(seed)
    times = np.cumsum(rng.exponential(gap, n))
    return [request(k+1, float(t), int(rng.integers(shards))) for k, t in enumerate(times)]


class EquivalenceTests(unittest.TestCase):
    """With the default switches, queue_run must reproduce predictive.run event for event."""
    def run_both(self, requests, full, seconds, budget, deadline, overhead=0.):
        out = []
        for run in (predictive.run, orchestration.queue_run):
            clock = Clock()
            out.append(run(requests, ShardTwin(full, overhead), executor(clock, seconds), budget, deadline,
                           clock=clock, sleep=clock.sleep))
        return out

    def test_queue_run_defaults_equal_predictive_run(self):
        cases = [([request(1, 0)], {}, {}, 2, 5),                                   # single request
                 ([request(1, 0), request(2, .25)], {}, {}, 1, 5),                  # budget pooling wait
                 ([request(1, 0), request(2, 1)], {0: 4.}, {0: 4.}, 2, 5)]          # downgrade to fixed
        for seed in range(25):                                                      # random multi-shard streams
            full = {0: 1.5, 1: 3., 2: 5.}
            seconds = {s: v*(1+.3*np.sin(seed+s)) for s, v in full.items()}
            cases.append((stream(seed), full, seconds, 1.+seed % 3, 4.+seed % 4))
        for requests, full, seconds, budget, deadline in cases:
            old, new = self.run_both(requests, full, seconds, budget, deadline, overhead=.1)
            self.assertEqual(old, new)

    def test_choose_defaults_equal_predictive_choose(self):
        rng = np.random.default_rng(4)
        for _ in range(200):
            n = int(rng.integers(1, 5))
            jobs = {}
            for s in range(n):
                full = float(rng.uniform(1, 6))
                jobs[s] = ([request(10*s+k, float(rng.uniform(0, 3)), s) for k in range(int(rng.integers(1, 4)))],
                           {'shard_full': prediction(full, .2), 'shard_fixed': prediction(full/2, .2)})
                jobs[s][0].sort(key=lambda r: r['time'])
            now, budget, deadline = float(rng.uniform(0, 4)), float(rng.uniform(.5, 4)), float(rng.uniform(2, 9))
            self.assertEqual(orchestration.choose(jobs, now, budget, deadline),
                             predictive.choose(jobs, now, budget, deadline))


class AblationTests(unittest.TestCase):
    def test_edf_order_plans_policies_only_and_keeps_the_never_idle_rule(self):
        fs = {'shard_full': prediction(2.), 'shard_fixed': prediction(1.)}
        jobs = {i: ([request(i+1, 3-i, i)], fs) for i in range(4)}        # shard 3 holds the oldest request
        got = orchestration.choose(jobs, 3, 5, 20, order='edf')
        self.assertEqual(got['plans_evaluated'], 16)
        self.assertEqual(got['order'], [3, 2, 1, 0])
        tight = {0: ([request(1, 0, 0)], {'shard_full': prediction(6.), 'shard_fixed': prediction(3.)}),
                 1: ([request(2, 1, 1)], {'shard_full': prediction(2.), 'shard_fixed': prediction(1.)})}
        self.assertEqual(orchestration.choose(tight, 1, 2, 10, order='edf')['shard'], 1)

    def test_no_overhead_plans_with_round_time_only(self):
        jobs = {0: ([request(1, 0)], {'shard_full': prediction(4., 3.), 'shard_fixed': prediction(2., 3.)})}
        self.assertEqual(orchestration.choose(jobs, 0, 10, 6)['policy'], 'shard_fixed')   # 4 + 3 > 6
        self.assertEqual(orchestration.choose(jobs, 0, 10, 6, overhead=False)['policy'], 'shard_full')

    def test_no_pooling_never_waits(self):
        requests = [request(1, 0), request(2, .25)]
        for pooling, first in ((True, [1, 2]), (False, [1])):
            clock = Clock()
            # fixed is forecast at 2 s > budget 1 for one request: the default controller pools
            events = orchestration.queue_run(requests, ShardTwin({0: 4.}), executor(clock), 1, 5, pooling=pooling,
                                             clock=clock, sleep=clock.sleep)
            self.assertEqual(events[0]['steps'], first)
            self.assertEqual(events[0]['started_seconds'], .25 if pooling else 0)


class BaselineTests(unittest.TestCase):
    def run_mode(self, requests, mode, budget=5, deadline=10, full=None, seconds=None, window=None):
        clock, calls = Clock(), []
        events = orchestration.baseline_run(requests, ShardTwin(full), executor(clock, seconds, calls), budget,
                                            deadline, mode, window=window, clock=clock, sleep=clock.sleep)
        return events, calls

    def test_every_mode_serves_each_request_once_and_summarizes(self):
        requests = stream(1, n=12)
        for mode in orchestration.BASELINES:
            events, _ = self.run_mode(requests, mode, budget=1.5, deadline=4, full={0: 1., 1: 2., 2: 3.},
                                      seconds={0: 1., 1: 2.5, 2: 3.}, window=1.)
            served = sorted(s for e in events for s in e['steps'])
            self.assertEqual(served, list(range(1, 13)), mode)
            for e in events:
                self.assertTrue(all(t <= e['started_seconds']+1e-12 for t in e['arrivals']), mode)
                np.testing.assert_allclose(e['latency_seconds'],
                                           [w+e['service_seconds'] for w in e['waiting_seconds']])
            starts = [e['started_seconds'] for e in events]
            readies = [e['model_ready_seconds'] for e in events]
            self.assertTrue(all(a >= b-1e-12 for a, b in zip(starts[1:], readies)), mode)  # one worker
            self.assertEqual(realtime.summarize(events)['requests'], 12)

    def test_immediate_serves_oldest_full_without_waiting(self):
        events, calls = self.run_mode([request(1, 0, 1), request(2, .5, 0), request(3, .6, 1)], 'immediate')
        self.assertEqual(calls, [([1], 'shard_full', 0.), ([2], 'shard_full', 2.), ([3], 'shard_full', 4.)])

    def test_window_serves_at_boundaries_only_what_arrived_by_them(self):
        events, calls = self.run_mode([request(1, .2), request(2, .5), request(3, 1.3)], 'window', window=1.)
        self.assertEqual([(c[0], c[2]) for c in calls], [([1, 2], 1.), ([3], 3.)])  # 3 waits for the boundary after the busy worker
        self.assertTrue(all(c[1] == 'shard_full' for c in calls))

    def test_lazy_starts_at_the_latest_full_start_and_batches(self):
        events, calls = self.run_mode([request(1, 0), request(2, 1)], 'lazy', deadline=10, full={0: 3.})
        self.assertEqual(calls, [([1, 2], 'shard_full', 7.)])  # 10 - 3

    def test_edf_downgrades_instead_of_waiting(self):
        events, calls = self.run_mode([request(1, 0, 0), request(2, .1, 1)], 'edf', budget=1, deadline=3,
                                      full={0: 4., 1: 1.}, seconds={0: 4., 1: 1.})
        self.assertEqual(calls[0], ([1], 'shard_fixed', 0.))   # full (4) misses budget 1 and deadline 3
        self.assertEqual(calls[1][1], 'shard_full')            # shard 1: 1 <= budget, finishes in time

    def test_invalid_arguments(self):
        with self.assertRaises(ValueError):
            orchestration.baseline_run([request(1, 0)], ShardTwin(), None, 1, 1, 'fifo')
        with self.assertRaises(ValueError):
            orchestration.baseline_run([request(1, 0)], ShardTwin(), None, 1, 1, 'window')
        with self.assertRaises(ValueError):
            orchestration.queue_run([request(1, 2), request(2, 1)], ShardTwin(), None, 1, 1)


class RunnerTests(unittest.TestCase):
    def test_blocks_cover_the_design_once_and_lock_hashes_ignore_line_endings(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.dt.run_orchestration import ARMS, blocks, lf_sha256
        cfg = json.loads(Path('configs/dt_orchestration.json').read_text())
        got = blocks(cfg)
        self.assertEqual(len(got), len(cfg['seeds'])*len(cfg['families'])*len(cfg['loads'])*len(cfg['environments']))
        self.assertEqual(len(set(got)), len(got))
        self.assertEqual(set(cfg['arms']), set(ARMS))
        self.assertEqual(set(cfg['c1_baselines']) | set(cfg['c2_ablations']) | {cfg['proposed']}, set(cfg['arms']))
        from scripts.dt.analyze_orchestration import utility_floors
        floors = utility_floors(cfg)
        self.assertEqual({floors[b] for b in ('immediate', 'window', 'lazy')}, {-.02})   # always-full baselines
        self.assertEqual({floors[b] for b in ('edf', 'rows_legacy')}, {-.005})          # same full/fixed space
        smoke = json.loads(Path('configs/dt_orchestration_smoke.json').read_text())
        self.assertEqual(set(utility_floors(smoke).values()), {-.005})                  # frozen before the change
        with tempfile.TemporaryDirectory() as tmp:
            a, b = Path(tmp)/'a.py', Path(tmp)/'b.py'
            a.write_bytes(b'x = 1\r\ny = 2\r\n'); b.write_bytes(b'x = 1\ny = 2\n')
            self.assertEqual(lf_sha256(a), lf_sha256(b))


class AnalyzerTests(unittest.TestCase):
    def test_holm_and_wilcoxon(self):
        from scripts.dt.analyze_orchestration import holm, wilcoxon
        np.testing.assert_allclose(holm([.01, .04, .03]), [.03, .06, .06])
        np.testing.assert_allclose(holm([.5, .001]), [.5, .002])
        self.assertEqual(wilcoxon([0, 0, 0], 'greater'), 1.)
        self.assertLess(wilcoxon([1, 2, 3, 1, 2, 3, 1, 2], 'greater'), .05)
        self.assertGreater(wilcoxon([-1, -2, -1, -3, -1, -2, -1, -2], 'greater'), .5)

    def test_audit_recomputes_the_arm_and_rejects_a_tampered_log(self):
        import json
        import tempfile
        from pathlib import Path
        from scripts.dt.analyze_orchestration import audit_arm
        requests = stream(2, n=8)
        clock = Clock()
        run = executor(clock, {0: 1., 1: 2., 2: 3.})
        def execute(batch, policy, started):
            return {**run(batch, policy, started), 'exact': True, 'structure_ok': True}
        events = orchestration.baseline_run(requests, ShardTwin(), execute, 1.5, 4, 'edf', clock=clock, sleep=clock.sleep)
        summary = {**realtime.summarize(events), 'bit_exact': True}
        with tempfile.TemporaryDirectory() as tmp:
            arm = Path(tmp)
            (arm/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events)+'\n')
            (arm/'final.json').write_text(json.dumps(dict(summary=summary)))
            self.assertEqual(audit_arm(arm, requests), summary['joint_success_requests'])
            (arm/'events.jsonl').write_text('\n'.join(json.dumps(e) for e in events[1:])+'\n')
            with self.assertRaisesRegex(ValueError, 'exactly once'):
                audit_arm(arm, requests)


class IntegrationTests(unittest.TestCase):
    def test_every_arm_ends_exact_on_the_same_deletions(self):
        import contextlib
        import io
        import json
        import tempfile
        from pathlib import Path
        import torch
        from scripts.dt.run_orchestration import run_arm
        from src.data.federated import partition
        from src.fu.shard_exact import ShardedUnlearner, assign_shards
        from test_shard_exact import config, synthetic
        threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)
        data = synthetic(800)
        parts = partition(data['train']['y'], 3, 5., 8)
        base = ShardedUnlearner(data['train'], data['validation'], parts, assign_shards(2, 8), 11, 'shard_full', config())
        fit = base.fit()
        cfg = json.loads(Path('configs/dt_orchestration.json').read_text())
        requests = []
        for step, client in enumerate([0, 1, 2, 3, 0], 1):
            shard = next(j for j, cs in enumerate(base.assignment) if client in cs)
            rows = np.asarray(parts[client][:2] if step < 5 else parts[client][-2:])  # step 5: client 0 again, other rows
            requests.append(dict(step=step, time=(step-1)*.005, client=client, shard=shard, rows=2, f=rows))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            results = [run_arm(base, fit, requests, name, 'none', cfg, 0, 1., 1., .01, data['calibration'], Path(tmp)/name)
                       for name in cfg['arms']]
        self.assertTrue(all(r['bit_exact'] and r['exact_all_events'] for r in results))
        self.assertEqual({r['alive_rows'] for r in results}, {790})
        self.assertTrue(all(r['requests'] == 5 for r in results))
        self.assertTrue(all(np.isfinite(r['calibration_macro_f1']) and np.isfinite(r['threshold_final']) for r in results))
        self.assertEqual(int(base.alive.sum()), 800)


if __name__ == '__main__':
    unittest.main()
