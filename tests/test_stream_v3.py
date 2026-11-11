import unittest

import numpy as np

from scripts.shard.shard_v2 import assign
from scripts.shard.stream_v3 import (FAMILIES, STREAM, arrival_times, backfill_plan, batch_windows, cost_plan, plan_stream,
                                     request_sizes, select_rows, zipf_ranking)

ROWS = [4768, 1308, 6095, 6102, 3364, 4508, 2490, 12280, 1056, 4445,
        29317, 4278, 15035, 7034, 814, 1794, 85, 5144, 317, 4050]     # partition 42, 20 clients (v2 meta.json)


class StreamPlanTests(unittest.TestCase):
    def test_deterministic_and_sizes_times_shared_by_families(self):
        A = assign(4, 20)
        for fam in FAMILIES:
            a, b = plan_stream(fam, ROWS, A), plan_stream(fam, ROWS, A)
            self.assertEqual(a, b)
            self.assertEqual([r['requested'] for r in a], request_sizes())
            self.assertEqual([r['time'] for r in a], arrival_times())
        self.assertTrue(all(np.diff(arrival_times()) > 0))

    def test_sample_level_cap_keeps_half_of_every_client(self):
        for S in (4, 5):
            A = assign(S, 20)
            for fam in FAMILIES:
                st = plan_stream(fam, ROWS, A); lost = np.zeros(20, int)
                for r in st:
                    self.assertEqual(r['kind'], 'sample'); self.assertGreaterEqual(r['rows'], 1)
                    self.assertLessEqual(r['rows'], r['requested']); lost[r['client']] += r['rows']
                self.assertTrue((lost <= np.floor(.5*np.array(ROWS))).all())

    def test_spread_visits_shards_in_turn_and_concentrated_stays_on_largest(self):
        for S in (4, 5):
            A = assign(S, 20)
            self.assertEqual([r['shard'] for r in plan_stream('spread', ROWS, A)], [k % S for k in range(STREAM['K'])])
            largest = int(np.argmax([sum(ROWS[c] for c in a) for a in A]))
            self.assertEqual({r['shard'] for r in plan_stream('concentrated', ROWS, A)}, {largest})

    def test_shard_label_matches_assignment_and_neutral_families_do_not_depend_on_S(self):
        for fam in ('proportional', 'poisson_zipf'):
            c4 = [r['client'] for r in plan_stream(fam, ROWS, assign(4, 20))]
            c5 = [r['client'] for r in plan_stream(fam, ROWS, assign(5, 20))]
            self.assertEqual(c4, c5)
        A = assign(5, 20)
        for r in plan_stream('poisson_zipf', ROWS, A): self.assertIn(r['client'], A[r['shard']])

    def test_zipf_ranking_is_a_permutation(self):
        self.assertEqual(sorted(zipf_ranking(20)), list(range(1, 21)))

    def test_exhausted_pool_raises(self):
        with self.assertRaises(ValueError): plan_stream('concentrated', [1]*20, assign(4, 20))   # cap floor(0.5) = 0
        with self.assertRaises(ValueError): plan_stream('nope', ROWS, assign(4, 20))


class RowsBatchesCostTests(unittest.TestCase):
    def test_select_rows_seeded_subset_of_alive(self):
        alive = np.arange(100, 200)
        a, b = select_rows(alive, 10, 3), select_rows(alive[::-1], 10, 3)
        np.testing.assert_array_equal(a, b); self.assertTrue(np.isin(a, alive).all()); self.assertEqual(len(np.unique(a)), 10)
        self.assertFalse(np.array_equal(a, select_rows(alive, 10, 4)))
        with self.assertRaises(ValueError): select_rows(alive, 101, 1)

    def test_batch_windows(self):
        t = [.3, 1.9, 2.0, 4.1]
        self.assertEqual(batch_windows(t, 0), t)
        self.assertEqual(batch_windows(t, 2), [2., 2., 4., 6.])
        self.assertTrue(all(p >= x for p, x in zip(batch_windows(t, 5), t)))

    def test_cost_plan_immediate_and_batched(self):
        A = [[0, 1], [2, 3]]; rows = [10, 10, 10, 10]
        st = [{'time': .5, 'client': 0, 'shard': 0, 'rows': 2}, {'time': 1.5, 'client': 1, 'shard': 0, 'rows': 3},
              {'time': 2.5, 'client': 2, 'shard': 1, 'rows': 1}]
        c0 = cost_plan(st, A, rows, rounds=10, W=0)
        self.assertEqual(c0['shard_rows_x_rounds'], (18 + 15 + 19)*10); self.assertEqual(c0['shard_retrains'], 3)
        self.assertEqual(c0['single_rows_x_rounds'], (38 + 35 + 34)*10); self.assertEqual(c0['latency_max'], 0)
        c2 = cost_plan(st, A, rows, rounds=10, W=2)                   # window [0,2): both shard-0 requests, one retrain
        self.assertEqual(c2['shard_rows_x_rounds'], (15 + 19)*10); self.assertEqual(c2['shard_retrains'], 2)
        self.assertEqual(c2['batches'], 2); self.assertAlmostEqual(c2['latency_max'], 1.5)
        self.assertEqual(c2['deleted_rows'], 6)


    def test_backfill_plan_follows_v2_rule(self):
        # shard 0 needs 20 rounds at 7 per request: back after the 3rd request; a repeat request restarts it
        self.assertEqual(backfill_plan([0, 1, 1], 2), [[1], [], [0]])
        self.assertEqual(backfill_plan([0, 0, 0, 0], 2), [[1]]*4)
        # round robin over 4 shards: the queue grows by 13 rounds per request and the ensemble empties at request 5
        self.assertEqual([len(i) for i in backfill_plan([k % 4 for k in range(6)], 4)], [3, 2, 2, 1, 0, 0])


if __name__ == '__main__':
    unittest.main()
