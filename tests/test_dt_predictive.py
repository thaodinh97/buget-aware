import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from src.dt import predictive, realtime
from test_dt_realtime import Clock, Twin, request
from test_shard_exact import TorchCase, config, synthetic


def prediction(value):
    return dict(mean_round=value, upper_round=value, service_upper=value, margin=1., experts=[value], weights=[1.])


class FakeTwin:
    def __init__(self): self.observations = []
    def forecast(self, shard, batch, rounds): return prediction(rounds/5)
    def observe(self, shard, result, forecast, seconds): self.observations.append(seconds)


class QueueTests(unittest.TestCase):
    def test_affordable_fixed_does_not_wait_until_deadline(self):
        stream = [request(1, 0)]
        def execute(clock):
            def run(batch, policy, started):
                clock.sleep(2.2)
                return dict(round_seconds=2., rows_x_rounds=10)
            return run
        old_clock, new_clock = Clock(), Clock()
        old = realtime.run(stream, Twin(full=4), execute(old_clock), 2, 5, padding=.05,
                           clock=old_clock, sleep=old_clock.sleep)
        new = predictive.run(stream, FakeTwin(), execute(new_clock), 2, 5,
                             clock=new_clock, sleep=new_clock.sleep)
        self.assertEqual(old[0]['deadline_misses'], 1)
        self.assertEqual(new[0]['deadline_misses'], 0)
        self.assertEqual(new[0]['started_seconds'], 0)
        self.assertEqual(new[0]['policy'], 'shard_fixed')

    def test_lookahead_preserves_more_full_requests_without_losing_deadlines(self):
        forecasts = {'shard_full': prediction(4.), 'shard_fixed': prediction(2.)}
        jobs = {0: ([request(1, 0, 0)], forecasts),
                1: ([request(i, 0, 1) for i in (2, 3, 4)], forecasts)}
        result = predictive.choose(jobs, 0, 10, 6)
        self.assertEqual(result['predicted_joint_success'], 4)
        self.assertEqual(result['shard'], 1)
        self.assertEqual(result['policy'], 'shard_full')
        self.assertEqual(result['policies'][1], 'shard_fixed')

    def test_no_future_arrivals_in_execution_and_feedback_only_after_completion(self):
        clock, twin = Clock(), FakeTwin()
        calls = []
        def execute(batch, policy, started):
            calls.append(([r['step'] for r in batch], len(twin.observations)))
            clock.sleep(2.)
            return dict(round_seconds=2., rows_x_rounds=10)
        events = predictive.run([request(1, 0), request(2, 1)], twin, execute, 2, 5,
                                clock=clock, sleep=clock.sleep)
        self.assertEqual(calls, [([1], 0), ([2], 1)])
        self.assertEqual(events[1]['waiting_seconds'], [1.])
        self.assertEqual(events[1]['latency_seconds'], [3.])

    def test_infeasible_budget_is_not_silently_dropped(self):
        clock, twin = Clock(), FakeTwin()
        def execute(batch, policy, started):
            clock.sleep(2.)
            return dict(round_seconds=2., rows_x_rounds=10)
        events = predictive.run([request(1, 0)], twin, execute, .1, 1,
                                clock=clock, sleep=clock.sleep)
        self.assertEqual(len(events), 1)
        self.assertTrue(events[0]['forced_over_budget'])
        self.assertEqual(events[0]['deadline_misses'], 1)

    def test_budget_pooling_wait_is_bounded_and_can_accept_new_arrivals(self):
        clock, twin = Clock(), FakeTwin()
        def execute(batch, policy, started):
            clock.sleep(2.)
            return dict(round_seconds=2., rows_x_rounds=10)
        events = predictive.run([request(1, 0), request(2, .25)], twin, execute, 1, 5,
                                clock=clock, sleep=clock.sleep)
        self.assertEqual(events[0]['steps'], [1, 2])
        self.assertEqual(events[0]['started_seconds'], .25)
        self.assertEqual(events[0]['deadline_misses'], 0)

    def test_enumeration_is_bounded(self):
        fs = {'shard_full': prediction(2.), 'shard_fixed': prediction(1.)}
        jobs = {i: ([request(i+1, 0, i)], fs) for i in range(4)}
        self.assertEqual(predictive.choose(jobs, 0, 5, 20)['plans_evaluated'], 384)
        with self.assertRaises(ValueError):
            predictive.choose({**jobs, 4: ([request(5, 0, 4)], fs)}, 0, 5, 20)

    def test_unaffordable_shard_cannot_block_an_affordable_one(self):
        jobs = {0: ([request(1, 0)], {'shard_full': prediction(6.), 'shard_fixed': prediction(3.)}),
                1: ([request(2, 1, 1)], {'shard_full': prediction(2.), 'shard_fixed': prediction(1.)})}
        result = predictive.choose(jobs, 1, 2, 10)
        self.assertEqual(result['shard'], 1)


class LearningTests(unittest.TestCase):
    def test_weights_use_pre_update_error_and_service_overhead_is_learned(self):
        rounds = [dict(shard=0, round=1, samples=10, wall_seconds=1., server_seconds=.2),
                  dict(shard=1, round=1, samples=100, wall_seconds=5., server_seconds=.2)]
        clients = [dict(shard=0, round=1, client_id=0, samples=10, compute_seconds=.8),
                   dict(shard=1, round=1, client_id=1, samples=100, compute_seconds=4.8)]
        fit = dict(round_costs=rounds, client_costs=clients, cost={'payload_pack_seconds': .02})
        twin = predictive.ServiceTwin(fit, lambda shard, batch: {0: 10})
        initial = twin.weights().copy()
        forecast = twin.forecast(0, [], 1)
        np.testing.assert_array_equal(twin.weights(), initial)
        twin.observe(0, dict(round_seconds=1., round_costs=rounds[:1], client_costs=clients[:1]), forecast, 3.)
        self.assertLess(twin.weights()[-1], initial[-1])
        after = twin.forecast(0, [], 1)
        self.assertGreater(after['overhead_seconds'], forecast['overhead_seconds'])
        self.assertGreaterEqual(after['margin'], 1.)
        self.assertAlmostEqual(sum(after['weights']), 1.)


class IntegrationTests(TorchCase):
    def test_new_twin_rows_ablation_and_legacy_end_exact_on_same_deletions(self):
        from scripts.dt.run_predictive_confirmation import run_arm
        from src.data.federated import partition
        from src.fu.shard_exact import ShardedUnlearner, assign_shards
        data = synthetic(800)
        parts = partition(data['train']['y'], 3, 5., 8)
        base = ShardedUnlearner(data['train'], data['validation'], parts, assign_shards(2, 8),
                               11, 'shard_full', config())
        fit = base.fit()
        cfg = json.loads(Path('configs/dt_predictive_confirmation.json').read_text())
        requests = []
        for step, client in enumerate([0, 1, 2, 3], 1):
            shard = next(j for j, cs in enumerate(base.assignment) if client in cs)
            requests.append(dict(step=step, time=(step-1)*.005, client=client, shard=shard,
                                 rows=2, f=np.asarray(parts[client][:2])))
        with tempfile.TemporaryDirectory() as tmp, contextlib.redirect_stdout(io.StringIO()):
            results = [run_arm(base, fit, requests, name, 'none', cfg, 0, 1., 1., Path(tmp)/name)
                       for name in cfg['predictors']]
        self.assertTrue(all(r['bit_exact'] and r['exact_all_events'] for r in results))
        self.assertTrue(all(r['alive_rows'] == 792 and r['requests'] == 4 for r in results))
        self.assertEqual(int(base.alive.sum()), 800)


if __name__ == '__main__':
    unittest.main()
