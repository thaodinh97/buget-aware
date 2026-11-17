import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from src.dt import realtime


class Clock:
    def __init__(self): self.now = 0.
    def __call__(self): return self.now
    def sleep(self, seconds): self.now += seconds


class Twin:
    def __init__(self, full=2.): self.full, self.seen, self.queries = full, [], []
    def forecast(self, shard, batch, rounds):
        self.queries.append([r['step'] for r in batch])
        return self.full * rounds / 20
    def factor(self): return 1.
    def observe(self, shard, result, forecast): self.seen.append(result['round_seconds'])


def request(step, time, shard=0):
    return dict(step=step, time=time, shard=shard, rows=1)


class TimingTests(unittest.TestCase):
    def execute(self, clock, actual):
        def go(batch, policy, started):
            clock.sleep(actual)
            return dict(round_seconds=actual, rows_x_rounds=20*len(batch))
        return go

    def test_arrivals_during_execution_wait_and_are_not_added_to_running_batch(self):
        clock, twin = Clock(), Twin()
        events = realtime.run([request(1, 0), request(2, 1)], twin, self.execute(clock, 2), 3, 10,
                              padding=0, clock=clock, sleep=clock.sleep)
        self.assertEqual([e['steps'] for e in events], [[1], [2]])
        self.assertEqual(events[1]['waiting_seconds'], [1])
        self.assertEqual(events[1]['latency_seconds'], [3])
        self.assertEqual(twin.seen, [2, 2])
        self.assertEqual(twin.queries[:2], [[1], [1]])

    def test_deferred_batch_reserves_service_time_but_underforecast_can_miss(self):
        clock, twin = Clock(), Twin(full=4)
        events = realtime.run([request(1, 0), request(2, 1)], twin, self.execute(clock, 3), 1, 5,
                              padding=0, clock=clock, sleep=clock.sleep)
        self.assertEqual(len(events), 1)
        event = events[0]
        self.assertEqual(event['steps'], [1, 2])
        self.assertEqual(event['policy'], 'shard_fixed')
        self.assertEqual(event['started_seconds'], 3)
        self.assertEqual(event['waiting_seconds'], [3, 2])
        self.assertEqual(event['latency_seconds'], [6, 5])
        self.assertEqual(event['deadline_misses'], 1)
        self.assertTrue(event['round_over_budget'])
        self.assertEqual(realtime.summarize(events)['joint_success_requests'], 0)

    def test_infeasible_request_is_executed_and_costed_not_silently_rejected(self):
        clock, twin = Clock(), Twin(full=4)
        events = realtime.run([request(1, 0)], twin, self.execute(clock, 2), .5, 1,
                              padding=0, clock=clock, sleep=clock.sleep)
        self.assertTrue(events[0]['forced_over_budget'])
        self.assertEqual(events[0]['deadline_misses'], 1)
        self.assertEqual(events[0]['rejected_requests'], 0)
        self.assertEqual(twin.seen, [2])

    def test_logging_time_delays_next_request_but_not_completed_request(self):
        clock, twin = Clock(), Twin()
        events = realtime.run([request(1, 0), request(2, 1)], twin, self.execute(clock, 2), 3, 10,
                              padding=0, clock=clock, sleep=clock.sleep, on_event=lambda _: clock.sleep(1))
        self.assertEqual(events[0]['model_ready_seconds'], 2)
        self.assertEqual(events[1]['started_seconds'], 3)
        self.assertEqual(events[1]['latency_seconds'], [4])

    def test_invalid_time_is_rejected(self):
        with self.assertRaises(ValueError):
            realtime.run([request(1, 2), request(2, 1)], None, None, 1, 1)


class StressTests(unittest.TestCase):
    def test_gradual_and_jitter_are_reproducible_and_not_a_single_step(self):
        from scripts.dt.run_online_confirmation import condition_at, stream_config
        cfg = json.loads(Path('configs/dt_online_confirmation.json').read_text())
        assignment = [[0, 1], [2, 3]]
        a = condition_at('gradual_clients', 3, 10, assignment, cfg, 0)
        b = condition_at('gradual_clients', 5, 10, assignment, cfg, 0)
        c = condition_at('gradual_clients', 9, 10, assignment, cfg, 0)
        self.assertLess(next(iter(a.slow.values())), next(iter(b.slow.values())))
        self.assertEqual(set(c.slow.values()), {4.})
        x = condition_at('jitter_bandwidth', 5.1, 10, assignment, cfg, 0)
        self.assertEqual(x, condition_at('jitter_bandwidth', 5.9, 10, assignment, cfg, 0))
        self.assertNotEqual(x, condition_at('jitter_bandwidth', 6.1, 10, assignment, cfg, 0))
        self.assertNotEqual(stream_config(cfg, 0)['seed_time'], stream_config(cfg, 1)['seed_time'])


from test_shard_exact import TorchCase, config, synthetic


class IntegrationTests(TorchCase):
    def test_three_online_arms_remove_same_rows_and_verify_exact(self):
        import numpy as np
        from scripts.dt.run_online_confirmation import run_arm
        from src.data.federated import partition
        from src.fu.shard_exact import ShardedUnlearner, assign_shards
        data = synthetic(800)
        parts = partition(data['train']['y'], 3, 5., 8)
        base = ShardedUnlearner(data['train'], data['validation'], parts, assign_shards(2, 8),
                               11, 'shard_full', config())
        fit = base.fit()
        cfg = json.loads(Path('configs/dt_online_confirmation.json').read_text())
        requests = []
        for step, client in enumerate([0, 1, 2, 3], 1):
            shard = next(j for j, clients in enumerate(base.assignment) if client in clients)
            requests.append(dict(step=step, time=(step-1)*.005, client=client, shard=shard,
                                 rows=2, f=np.asarray(parts[client][:2])))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            rows = [run_arm(base, fit, requests, name, 'none', cfg, 0, 1, 1, Path(tmp)/name)
                    for name in cfg['predictors']]
        self.assertTrue(all(r['bit_exact'] and r['exact_all_events'] for r in rows))
        self.assertTrue(all(r['requests'] == 4 and r['alive_rows'] == 792 for r in rows))
        self.assertEqual(int(base.alive.sum()), 800)


if __name__ == '__main__':
    unittest.main()
