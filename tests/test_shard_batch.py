"""Batch deletion preserves exact retraining while charging each touched shard once.

Written for the v3 branch's own unlearn_batch; adapted on 2026-11-18 to the single unlearn_batch kept when the DT branch
was merged (docs/shard_exact_merge.md): result kind 'batch', details['shards_retrained'], two extra history keys, the
unlearner's policy as default (shard_full / shard_fixed only), client-level requests allowed."""
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import torch

from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner

BATCH_KEYS = ('batch_first_step', 'batch_size')   # history keys only unlearn_batch writes


class BatchTests(unittest.TestCase):
    def setUp(self):
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        rng = np.random.default_rng(42)

        def split(n, offset):
            return {'X': rng.normal(size=(n, 4)).astype(np.float32),
                    'y': np.tile([0, 1], n//2), 'row_id': np.arange(offset, offset + n)}

        self.train, self.validation = split(320, 0), split(80, 1000)
        self.parts = list(np.arange(320).reshape(8, 40))
        self.assignment = [[0, 1, 2, 3], [4, 5, 6, 7]]
        self.cfg = ShardConfig(fl={'rounds': 3, 'local_epochs': 1, 'batch_size': 128, 'lr': .01},
                               shard_fixed_rounds=1, backfill_rounds=1)

    def make(self, policy='shard_full', measure_payload=False):
        return ShardedUnlearner(self.train, self.validation, self.parts, self.assignment, 17,
                                policy, self.cfg, measure_payload=measure_payload)

    def assert_models_equal(self, left, right):
        self.assertEqual(left.included(), right.included())
        for j in left.included():
            a, b = left.shards[j], right.shards[j]
            self.assertTrue(np.array_equal(a.idx, b.idx))
            self.assertEqual(a.rounds, b.rounds)
            self.assertTrue(all(torch.equal(x, y) for x, y in zip(a.model.state_dict().values(),
                                                               b.model.state_dict().values())))

    def test_coalesces_touched_shards_and_matches_sequential_final_state(self):
        base = self.make()
        base.fit()
        batch, immediate = base.fork(), base.fork()
        requests = [ForgetRequest(0, self.parts[0][:7]), ForgetRequest(1, self.parts[1][:9]),
                    ForgetRequest(0, self.parts[0][7:12]), ForgetRequest(5, self.parts[5][:6])]
        got = batch.unlearn_batch(requests)
        sequential = [immediate.unlearn(request) for request in requests]
        self.assert_models_equal(batch, immediate)
        self.assertTrue(batch.verify_bit_exact()['bit_exact'])
        self.assertEqual(got.details['shards_retrained'], [0, 1])
        self.assertEqual(got.details['batch_size'], 4)
        self.assertEqual((got.step, got.rows, got.kind, got.exactness_scope), (4, 27, 'batch', 'unconditional'))
        self.assertEqual(got.client, -1)
        self.assertTrue(got.exact and got.structure_ok)
        self.assertEqual(got.cost['rounds'], 6)
        self.assertEqual(got.cost['rows_x_rounds'], (320 - 27)*3)
        self.assertLess(got.cost['rows_x_rounds'], sum(r.cost['rows_x_rounds'] for r in sequential))
        self.assertEqual(sum(r['samples'] for r in got.round_costs), got.cost['rows_x_rounds'])
        self.assertEqual([{k: v for k, v in r.items() if k not in BATCH_KEYS} for r in batch.history], immediate.history)
        self.assertEqual([(r['batch_first_step'], r['batch_size']) for r in batch.history], [(1, 4)]*4)
        self.assertEqual(batch.assignment, self.assignment)
        self.assertEqual(batch.present, set(range(8)))
        self.assertTrue(base.alive.all())
        self.assertEqual(base.step, 0)
        self.assertNotIn('STRUCTURE_CHANGED', {w['code'] for w in got.structured_warnings})
        json.dumps(got.to_json())

    def test_invalid_later_request_does_not_apply_valid_prefix(self):
        fed = self.make()
        fed.fit()
        original = fed.fork()
        first = ForgetRequest(0, self.parts[0][:4])
        invalid = [ForgetRequest(99, [0]), ForgetRequest(1, []),
                   ForgetRequest(1, [40, 40]), ForgetRequest(1, [80]),
                   ForgetRequest(0, self.parts[0][2:5])]
        for request in invalid:
            with self.subTest(request=request), self.assertRaises(ValueError):
                fed.unlearn_batch([first, request])
            self.assertTrue(fed.alive.all())
            self.assertEqual(fed.step, 0)
            self.assertEqual(fed.history, [])
            self.assertEqual(fed.totals, original.totals)
            self.assert_models_equal(fed, original)
        fed.unlearn(first)
        with self.assertRaises(ValueError):
            fed.unlearn_batch([first])
        self.assertEqual(fed.step, 1)

    def test_requires_fit_nonempty_batch_and_no_pending_backlog(self):
        fed = self.make('drop_backfill')
        request = ForgetRequest(0, self.parts[0][:3])
        with self.assertRaisesRegex(RuntimeError, 'fit'):
            fed.unlearn_batch([request])
        fed.fit()
        with self.assertRaisesRegex(ValueError, 'at least one'):
            fed.unlearn_batch([], 'shard_full')
        fed.unlearn(request)
        with self.assertRaisesRegex(RuntimeError, 'backlog'):
            fed.unlearn_batch([ForgetRequest(1, self.parts[1][:3])], 'shard_full')
        fed.finish_backlog()
        with self.assertRaisesRegex(ValueError, 'supports'):
            fed.unlearn_batch([ForgetRequest(1, self.parts[1][:3])])
        self.assertTrue(fed.unlearn_batch([ForgetRequest(1, self.parts[1][:3])], 'shard_full').exact)

    def test_one_request_matches_unlearn_with_explicit_full_rounds(self):
        base = self.make('utility_stop')
        base.fit()
        batch, single = base.fork(), base.fork('shard_full')
        request = ForgetRequest(3, self.parts[3][:5])
        with self.assertRaisesRegex(ValueError, 'supports'):
            batch.unlearn_batch([request])
        got, expected = batch.unlearn_batch([request], 'shard_full'), single.unlearn(request)
        self.assert_models_equal(batch, single)
        self.assertEqual(got.cost['rounds'], self.cfg.fl['rounds'])
        self.assertEqual(got.cost['cumulative_unlearning'], expected.cost['cumulative_unlearning'])
        self.assertEqual(got.forget_request_hash, expected.forget_request_hash)
        self.assertEqual(got.policy, 'shard_full')
        self.assertEqual(batch.policy, 'utility_stop')

    def test_checkpoint_continues_batches_and_single_requests(self):
        live = self.make(measure_payload=True)
        live.fit()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'batch'
            got = live.unlearn_batch([ForgetRequest(0, self.parts[0][:5]),
                                      ForgetRequest(4, self.parts[4][:5])], checkpoint_dir=path)
            self.assertEqual(got.final_checkpoint, path.as_posix())
            restored = ShardedUnlearner.load(path, self.train, self.validation, self.parts, measure_payload=False)
            self.assertEqual(restored.history, live.history)
            self.assertEqual(restored.step, 2)
            self.assertEqual(restored.totals, live.totals)
            self.assert_models_equal(live, restored)
            requests = [ForgetRequest(1, self.parts[1][:6]), ForgetRequest(0, self.parts[0][5:9])]
            a, b = live.unlearn_batch(requests), restored.unlearn_batch(iter(requests))
            self.assertEqual(a.cost['cumulative_unlearning'], b.cost['cumulative_unlearning'])
            self.assert_models_equal(live, restored)
            request = ForgetRequest(6, self.parts[6][:3])
            live.unlearn(request)
            restored.unlearn(request)
            self.assert_models_equal(live, restored)
            self.assertTrue(restored.verify_bit_exact()['bit_exact'])


if __name__ == '__main__':
    unittest.main()
