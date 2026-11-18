import unittest

import numpy as np

from scripts.shard.fpr_threshold_check import bayes, choose, fpr_at_recall, sweep
from src.eval.utility_breakdown import detector_metrics


class ThresholdRuleTests(unittest.TestCase):
    def setUp(self):
        rng = np.random.default_rng(3)
        self.y = rng.integers(0, 2, 400)
        self.m = np.round(rng.normal(size=400) + 1.5*self.y - .5, 1)  # rounded: many ties

    def test_sweep_matches_brute_force(self):
        s = sweep(self.m, self.y)
        for k in range(0, len(s['t']), 7):
            got = detector_metrics(self.y, self.m, s['t'][k])
            self.assertAlmostEqual(s['fpr'][k], got['fpr'])
            self.assertAlmostEqual(s['tpr'][k], got['recall_attack'])
            self.assertAlmostEqual(s['macro_f1'][k], got['macro_f1'])

    def test_rules_respect_their_definition(self):
        s = sweep(self.m, self.y)
        self.assertEqual(choose('default', s), 0.)
        best = detector_metrics(self.y, self.m, choose('max_f1', s))['macro_f1']
        self.assertAlmostEqual(best, s['macro_f1'].max())
        for rule, q in (('fpr10', .10), ('fpr05', .05)):
            t = choose(rule, s)
            got = detector_metrics(self.y, self.m, t)
            self.assertLessEqual(got['fpr'], q)
            lower = s['t'][s['t'] < t]
            if len(lower):  # the next lower threshold breaks the FPR constraint
                self.assertGreater(detector_metrics(self.y, self.m, lower.max())['fpr'], q)

    def test_fpr_at_recall_reaches_recall(self):
        s = sweep(self.m, self.y)
        for r in (.95, .99):
            fpr = fpr_at_recall(s, r)
            k = np.flatnonzero(s['fpr'] == fpr)
            self.assertTrue((s['tpr'][k] >= r).any())

    def test_bayes_counts_conflicting_vectors(self):
        X = np.array([[0, 0], [0, 0], [0, 0], [1, 1], [2, 2]], np.float32)
        y = np.array([1, 1, 0, 0, 1])
        b = bayes({'X': X, 'y': y})
        self.assertEqual(b['distinct_vectors'], 3)
        self.assertEqual(b['vectors_with_both_labels'], 1)
        self.assertAlmostEqual(b['bayes_error'], 1/5)
        self.assertAlmostEqual(b['fpr_floor_majority'], 1/2)  # the normal row of [0, 0] is outvoted


class Log1pVariantTests(unittest.TestCase):
    def test_recovers_raw_values_and_restandardizes_on_validation(self):
        from scripts.shard.fpr_ceiling_check import log1p_variant
        rng = np.random.default_rng(5)
        num = ['duration', 'src_bytes']
        state = {'numeric': num, 'feature_names': num + ['proto_tcp'],
                 'mean': {'duration': 7.3, 'src_bytes': 6.1e5}, 'scale': {'duration': 63.8, 'src_bytes': 2.4e7}}

        def split(n):
            raw = np.c_[rng.exponential(5, n), rng.integers(0, 10**6, n)]
            z = (raw - [7.3, 6.1e5])/[63.8, 2.4e7]
            return raw, {'X': np.c_[z, rng.integers(0, 2, n)].astype(np.float32), 'y': rng.integers(0, 2, n)}

        raws, data = {}, {}
        for name in ('train', 'validation', 'calibration'):
            raws[name], data[name] = split(300)
        out = log1p_variant(data, state)
        logs_va = np.log1p(np.c_[raws['validation'][:, 0], raws['validation'][:, 1]])
        mu, sd = logs_va.mean(0), logs_va.std(0)
        for name in data:
            expected = (np.log1p(raws[name]) - mu)/sd
            np.testing.assert_allclose(out[name]['X'][:, 1], expected[:, 1], atol=1e-5)  # integer column, exact after rounding
            np.testing.assert_allclose(out[name]['X'][:, 0], expected[:, 0], atol=1e-3)  # float column, float32 precision
            np.testing.assert_array_equal(out[name]['X'][:, 2], data[name]['X'][:, 2])
            np.testing.assert_array_equal(out[name]['y'], data[name]['y'])


if __name__ == '__main__':
    unittest.main()
