"""scripts/shard/dt_ablation.py: the with-DT / without-DT admission pilot, on synthetic data."""
from unittest.mock import patch

import numpy as np
import torch

from scripts.shard import dt_ablation as ab
from src.fu.shard_exact import ForgetRequest, ShardedUnlearner, _Trainer, assign_shards
from src.data.federated import partition
from test_shard_exact import TRAIN_CFG, TorchCase, config, synthetic, weights_equal


def ensemble(data, clients=8, shards=2, seed=5):
    parts = partition(data['train']['y'], 42, .5, clients)
    fed = ShardedUnlearner(data['train'], data['validation'], parts, assign_shards(shards, clients), seed, 'shard_full',
                           config(), measure_payload=False)
    fed.fit()
    return fed, parts


class ScenarioTests(TorchCase):
    def test_load_factors(self):
        self.assertTrue((ab.load_factors('stable', 20, 10, 301) == 1).all())
        step, burst = ab.load_factors('step', 20, 10, 301), ab.load_factors('burst', 20, 10, 301)
        self.assertTrue((step[:3] == 1).all())
        self.assertEqual(sorted(np.unique(step[3:]).tolist()), [1., 3.])
        self.assertTrue(((step[3:] == 3).sum(1) == 10).all())                 # half of the clients, every later request
        self.assertTrue((burst[3:6] == step[3:6]).all() and (burst[6:] == 1).all())
        drift = ab.load_factors('drift', 20, 10, 301)
        self.assertTrue(np.allclose(drift[0], 1) and np.allclose(drift[-1], 3))
        self.assertTrue((ab.load_factors('noisy', 20, 10, 301) == ab.load_factors('noisy', 20, 10, 301)).all())
        self.assertFalse((ab.load_factors('step', 20, 10, 301) == ab.load_factors('step', 20, 10, 302)).all())
        with self.assertRaises(ValueError):
            ab.load_factors('other', 20, 10, 301)


class InFlightTests(TorchCase):
    def test_full_retrain_passes_through_the_fixed_model(self):
        """Stopping a full retrain after shard_fixed_rounds gives shard_fixed; continuing gives shard_full, bit for bit."""
        fed, parts = ensemble(synthetic(1600))
        c = fed.assignment[1][0]
        request = ForgetRequest(c, parts[c][:5])
        fixed, full = fed.fork('shard_fixed'), fed.fork('shard_full')
        fixed.unlearn(request); full.unlearn(request)
        trainer = _Trainer(full, 1, full._shard_idx(full.assignment, 1), full.seed + 1000)
        for _ in range(fed.config.shard_fixed_rounds):
            trainer.step(None, None)
        self.assertTrue(weights_equal(trainer.m, fixed.shards[1].model))
        for _ in range(fed.r_base - fed.config.shard_fixed_rounds):
            trainer.step(None, None)
        self.assertTrue(weights_equal(trainer.m, full.shards[1].model))

    def test_slow_worker_keeps_the_weights(self):
        fed, parts = ensemble(synthetic(1600))
        c = fed.assignment[0][0]
        request = ForgetRequest(c, parts[c][:5])
        plain, slow = fed.fork('shard_full'), fed.fork('shard_full')
        slow.worker_factory = ab.slow_factory(slow, np.full(len(parts), 1.5))
        a, b = plain.unlearn(request), slow.unlearn(request)
        self.assertTrue(weights_equal(plain.shards[0].model, slow.shards[0].model))
        self.assertFalse(hasattr(fed, 'worker_factory'))
        ratio = sum(r['compute_seconds_sum'] for r in b.round_costs) / sum(r['compute_seconds_sum'] for r in a.round_costs)
        self.assertGreater(ratio, 1.)


class ExperimentTests(TorchCase):
    def run_experiment(self):
        data = synthetic(2400)
        with patch.dict(ab.CONFIGS, tiny=TRAIN_CFG):
            parts = partition(data['train']['y'], 42, .5, 8)
            stream = [{'step': 1, 'client': 0, 'kind': 'sample', 'f': parts[0][:4]},
                      {'step': 2, 'client': 5, 'kind': 'sample', 'f': parts[5][:4]},
                      {'step': 3, 'client': 0, 'kind': 'sample', 'f': parts[0][4:9]},
                      {'step': 4, 'client': 6, 'kind': 'sample', 'f': parts[6][:3]}]
            with patch.object(ab, 'request_stream', lambda y, p: stream):
                return ab.experiment(data, [5], [301], shards=2, train_config='tiny', clients=8, check=True, log=lambda m: None)

    def test_outcomes_are_consistent(self):
        decisions, trajectories, checks, info = self.run_experiment()
        cells = len(ab.BUDGETS) * len(ab.SCENARIOS)                             # one scenario seed
        self.assertEqual(len(decisions), 4 * cells * len(ab.ARMS))
        self.assertEqual(len(trajectories), cells * len(ab.ARMS))
        self.assertTrue(decisions.outcome.isin(['ok', 'violation', 'conservative']).all())
        late = decisions.spent_seconds > decisions.budget_seconds
        self.assertTrue(((decisions.outcome == 'violation') == late).all())
        timely = decisions[~late]
        self.assertTrue(((timely.outcome == 'ok') == (timely.rounds == timely.best_rounds)).all())
        self.assertTrue((timely[timely.outcome == 'conservative'].rounds < timely[timely.outcome == 'conservative'].best_rounds).all())
        oracle = decisions[decisions.arm == 'oracle']
        self.assertTrue((oracle.outcome == 'ok').all())
        self.assertTrue((decisions[decisions.arm == 'always_full'].policy == 'full').all())
        self.assertTrue((decisions[decisions.arm == 'always_fixed'].policy == 'fixed').all())
        self.assertFalse((decisions[decisions.arm == 'reactive'].policy == 'reject').any())
        # an arm without a forecast records none; the 10-round trace is a prefix of the 20-round one
        self.assertTrue(decisions[decisions.group == 'no_twin'].forecast_full_seconds.isna().all())
        self.assertTrue(decisions[decisions.group == 'profiler'].forecast_full_seconds.notna().all())
        self.assertTrue((decisions.actual_fixed_seconds < decisions.actual_full_seconds).all())
        # under stable load the telemetry twin is the static profiler
        stable = decisions[decisions.scenario == 'stable']
        self.assertEqual(stable[stable.arm == 'twin_fresh'].rounds.tolist(), stable[stable.arm == 'static'].rounds.tolist())
        self.assertTrue(checks.weights_equal.all())
        self.assertEqual(set(decisions.level), {'tight', 'moderate', 'loose'})
        # a shared factor of 1 leaves the static forecast: adaptive_global starts as the profiler
        first = decisions[decisions.step == 1]
        self.assertTrue(np.allclose(first[first.arm == 'adaptive_global'].forecast_full_seconds.values,
                                    first[first.arm == 'static'].forecast_full_seconds.values))
        self.assertEqual(len(info), 1)
        self.assertFalse(ab.summarise(decisions, ['arm']).isna().ok.any())

    def test_same_state_same_actual_seconds(self):
        """Arms that took the same decisions so far face the same measured trace (one retrain per state)."""
        decisions, *_ = self.run_experiment()
        first = decisions[decisions.step == 1]
        for _, g in first.groupby(['scenario', 'scenario_seed']):
            self.assertEqual(g.actual_full_seconds.nunique(), 1)
