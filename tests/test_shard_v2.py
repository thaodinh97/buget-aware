import os
from pathlib import Path
import tempfile
import unittest

import numpy as np
from scipy import stats

from scripts.shard.shard_v2 import check_design, data_content_sha256, noninferior, norm_rel, structure_plan


def req(step, client, kind='client'):
    return {'step': step, 'client': client, 'kind': kind}


class NoninferiorityTests(unittest.TestCase):
    def test_bound_is_one_sided_paired_t(self):
        d = np.array([-.004, -.006, -.002, -.008, -.005])
        t = stats.t.ppf(.95, 4); se = d.std(ddof=1)/np.sqrt(5)
        r = noninferior(d, .01, 'lower')
        self.assertAlmostEqual(r['bound'], d.mean() - t*se)
        self.assertTrue(r['ok'])
        self.assertFalse(noninferior(d - .006, .01, 'lower')['ok'])     # mean -0.011: lower bound below -0.01

    def test_higher_is_worse_uses_upper_bound(self):
        d = np.array([.001, .003, .002, .004])
        r = noninferior(d, .02, 'higher')
        self.assertAlmostEqual(r['bound'], d.mean() + stats.t.ppf(.95, 3)*d.std(ddof=1)/2)
        self.assertTrue(r['ok'])
        self.assertFalse(noninferior(d + .018, .02, 'higher')['ok'])

    def test_point_estimate_inside_margin_can_still_fail(self):
        d = np.array([.0, -.02, .0, -.02, .0])                           # mean -0.008, but noisy
        self.assertFalse(noninferior(d, .01, 'lower')['ok'])

    def test_identical_runs_pass_and_bad_input_raises(self):
        self.assertTrue(noninferior(np.zeros(5), .01, 'lower')['ok'])
        with self.assertRaises(ValueError): noninferior([.1], .01, 'lower')
        with self.assertRaises(ValueError): noninferior([.1, np.nan], .01, 'lower')
        with self.assertRaises(ValueError): noninferior([.1, .2], .01, 'sideways')


class DataHashTests(unittest.TestCase):
    def data(self):
        rng = np.random.default_rng(0)
        return {s: {'X': rng.normal(size=(5, 3)).astype(np.float32), 'y': np.arange(5), 'row_id': np.array(list('abcde'))}
                for s in ('train', 'validation', 'calibration')}

    def test_hash_depends_on_content_not_key_order_or_byte_order(self):
        a = self.data(); h = data_content_sha256(a)
        b = {s: dict(reversed(list(v.items()))) for s, v in a.items()}
        self.assertEqual(data_content_sha256(b), h)
        c = {s: {k: v.astype(v.dtype.newbyteorder('>')) for k, v in d.items()} for s, d in a.items()}
        self.assertEqual(data_content_sha256(c), h)

    def test_hash_changes_with_values_dtype_or_shape(self):
        a = self.data(); h = data_content_sha256(a)
        b = self.data(); b['train']['X'][0, 0] += 1
        self.assertNotEqual(data_content_sha256(b), h)
        c = self.data(); c['validation']['y'] = c['validation']['y'].astype(np.int32)
        self.assertNotEqual(data_content_sha256(c), h)
        e = self.data(); e['calibration']['X'] = e['calibration']['X'].reshape(3, 5)
        self.assertNotEqual(data_content_sha256(e), h)

    def test_unused_splits_are_ignored(self):
        a = self.data(); b = self.data(); b['test'] = {'X': np.ones(3)}
        self.assertEqual(data_content_sha256(a), data_content_sha256(b))


class PathTests(unittest.TestCase):
    def test_relative_inside_and_absolute_outside(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)/'repo'; (root/'data'/'x').mkdir(parents=True)
            self.assertEqual(norm_rel(root/'data'/'x', root), 'data/x')
            self.assertEqual(norm_rel(root, root), '.')
            self.assertEqual(norm_rel(Path(tmp)/'other', root), Path(os.path.abspath(Path(tmp)/'other')).as_posix())
            self.assertNotEqual(norm_rel(Path(tmp)/'repo2', root), 'repo2')          # a sibling with a common prefix is outside

    @unittest.skipUnless(os.name == 'nt', 'drive-letter case only exists on Windows')
    def test_drive_letter_case_is_ignored(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); sub = str(root/'a'/'b')
            swapped = (sub[0].lower() if sub[0].isupper() else sub[0].upper()) + sub[1:]
            self.assertEqual(norm_rel(swapped, root), 'a/b')


class StructureTests(unittest.TestCase):
    def test_design_minimum(self):
        self.assertEqual(check_design([[0, 1, 2, 3], [4, 5, 6, 7]], 4), [4, 4])
        with self.assertRaises(ValueError): check_design([[0, 1, 2, 3], [4, 5, 6]], 4)

    def test_merge_when_below_floor_into_smallest_lowest_index(self):
        A = [[0, 1, 2], [3, 4, 5, 6], [7, 8, 9]]
        p = structure_plan(A, [req(1, 0)], 3)[0]
        self.assertEqual(p['event'], 'merge_0_into_2')                            # shards 1 (4) and 2 (3): 2 has fewer
        self.assertEqual((p['retrain'], p['retired']), (2, [0]))
        self.assertEqual(p['assign_after'], [[], [3, 4, 5, 6], [1, 2, 7, 8, 9]])
        self.assertEqual(p['assign_before'], A)
        q = structure_plan([[0, 1, 2], [3, 4, 5], [6, 7, 8]], [req(1, 0)], 3)[0]
        self.assertEqual(q['event'], 'merge_0_into_1')                            # tie on size: lowest index

    def test_sample_requests_and_large_shards_only_retrain(self):
        A = [[0, 1, 2, 3], [4, 5, 6, 7]]
        plan = structure_plan(A, [req(1, 0, 'sample'), req(2, 0)], 3)
        self.assertEqual([(p['event'], p['retrain']) for p in plan], [('retrain', 0), ('retrain', 0)])
        self.assertEqual(plan[-1]['assign_after'], [[1, 2, 3], [4, 5, 6, 7]])

    def test_floor_holds_after_every_step(self):
        A = [[0, 1, 2, 3], [4, 5, 6, 7], [8, 9, 10, 11], [12, 13, 14, 15]]
        stream = [req(i + 1, c) for i, c in enumerate([0, 1, 4, 8, 5, 12, 9, 2])]
        for p in structure_plan(A, stream, 3):
            active = [a for a in p['assign_after'] if a]
            self.assertTrue(len(active) == 1 or min(map(len, active)) >= 3, p)
            self.assertEqual(sorted(c for a in active for c in a), sorted(c for a in p['assign_before'] for c in a if c != p['client']))

    def test_empty_shard_is_removed_without_merging(self):
        p = structure_plan([[0], [1, 2]], [req(1, 0)], 0)[0]
        self.assertEqual((p['event'], p['retrain'], p['retired']), ('removed_empty', None, [0]))


if __name__ == '__main__':
    unittest.main()
