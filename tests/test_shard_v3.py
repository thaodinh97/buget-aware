"""End-to-end synthetic smoke of every v3 family, policy and shard count.

Training seeds and stream sizes here are deliberately outside the frozen study.
The real training/component/evaluator path is exercised without claiming evidence.
"""
import contextlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd
import torch

from scripts.shard.evaluate_shard_v3 import evaluate
from scripts.shard.run_shard_stream_v3 import run_experiment
from scripts.shard.shard_v2 import assign
from scripts.shard.stream_v3 import FAMILIES, STREAM, backfill_plan, batch_windows, cost_plan
from src.fu.shard_exact import ShardConfig


class RunnerSmokeTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        threads = torch.get_num_threads()
        deterministic = torch.are_deterministic_algorithms_enabled()
        cls.addClassCleanup(torch.set_num_threads, threads)
        cls.addClassCleanup(torch.use_deterministic_algorithms, deterministic)
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        cls.temp = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.temp.cleanup)
        cls.root = Path(cls.temp.name)
        rng = np.random.default_rng(831)

        def split(n, offset):
            y = np.tile([0, 1], n//2)
            x = rng.normal(size=(n, 4)).astype(np.float32)
            x[:, 0] += 2*y - 1
            return {'X': x, 'y': y, 'row_id': np.arange(offset, offset + n)}

        data = {'train': split(800, 0), 'validation': split(120, 1000), 'calibration': split(120, 2000)}
        parts = list(np.arange(800).reshape(20, 40))
        cfg = ShardConfig(fl={'rounds': 3, 'local_epochs': 1, 'batch_size': 64, 'lr': .01},
                          shard_fixed_rounds=2, backfill_rounds=1, hidden=(8, 4))
        stream_cfg = {**STREAM, 'size_median': 2, 'size_sigma': .5}
        cls.steps, cls.exactness, cls.streams, cls.records = {}, {}, {}, {}
        for family in FAMILIES:
            path = cls.root/f'stream_{family}'
            with contextlib.redirect_stdout(io.StringIO()):
                cls.records[family] = run_experiment(path, family, data, parts, cfg, [7, 8],
                                                     stream_config=stream_cfg, measure_payload=False)
            cls.steps[family] = pd.read_csv(path/'steps.csv')
            cls.exactness[family] = pd.read_csv(path/'exactness.csv')
            cls.streams[family] = json.loads((path/'stream.json').read_text(encoding='utf-8'))

    def test_runner_and_evaluator_agree_on_exactness_and_plans(self):
        result = evaluate(self.root)
        self.assertTrue(result['gates']['G1']['pass'], result['gates']['G1'])
        self.assertTrue(result['gates']['G2']['pass'], result['gates']['G2'])
        self.assertFalse(result['confirmed_all'])
        self.assertTrue(all(r['completed'] and r['reproducible'] for r in self.records.values()))
        self.assertTrue(all(r['design_name'] == 'synthetic_smoke' for r in self.records.values()))

    def test_batch_arrivals_keep_pending_rows_until_window_end(self):
        for family in FAMILIES:
            frame = self.steps[family]
            for S in (4, 5):
                requests = self.streams[family]['streams'][str(S)]
                for W in (2, 5):
                    processing = batch_windows([r['time'] for r in requests], W)
                    for seed in (7, 8):
                        rows = frame[(frame.S == S) & (frame.seed == seed) & (frame.policy == f'batch_full_{W}')]
                        arrivals = rows[rows.kind == 'request'].sort_values('step')
                        ends = rows[rows.kind == 'batch_end']
                        self.assertEqual(len(arrivals), 20)
                        pending, last = 0, None
                        for (_, row), request, at in zip(arrivals.iterrows(), requests, processing):
                            if at != last:
                                pending = 0
                            pending += request['rows']
                            last = at
                            self.assertEqual(row.pending_rows, pending)
                            self.assertAlmostEqual(row.processed_at, at)
                            self.assertAlmostEqual(row.latency, at - request['time'])
                            self.assertAlmostEqual(row.exposure, request['rows']*row.latency)
                            self.assertEqual(row.delta_rows_x_rounds, 0)
                        self.assertEqual(len(ends), len(set(processing)))
                        self.assertTrue((ends.pending_rows == 0).all())
                        self.assertTrue(ends.exact.all() and ends.structure_ok.all())
                        self.assertAlmostEqual(ends.exposure.sum(), arrivals.exposure.sum())

    def test_measured_costs_and_final_models_match_metadata_plan(self):
        for family in FAMILIES:
            frame = self.steps[family]
            for S in (4, 5):
                requests = self.streams[family]['streams'][str(S)]
                for W in (0, 2, 5):
                    policy = 'shard_full' if W == 0 else f'batch_full_{W}'
                    planned = cost_plan(requests, assign(S, 20), [40]*20, 3, W)
                    for seed in (7, 8):
                        rows = frame[(frame.S == S) & (frame.seed == seed) & (frame.policy == policy)]
                        self.assertEqual(rows.iloc[-1].rows_x_rounds, planned['shard_rows_x_rounds'])
                        self.assertEqual(rows.delta_rows_x_rounds.sum(), planned['shard_rows_x_rounds'])
                        self.assertEqual(rows.delta_rounds.sum(), 3*planned['shard_retrains'])
            checked = self.exactness[family]
            batches = checked[checked.policy.str.startswith('batch_full_')]
            self.assertTrue(batches.batch_matches_shard_full.all())
            self.assertTrue(checked.bit_exact.all())

    def test_backfill_empty_ensemble_is_unavailable_and_nan(self):
        for family in FAMILIES:
            frame = self.steps[family]
            for S in (4, 5):
                requests = self.streams[family]['streams'][str(S)]
                expected = backfill_plan([r['shard'] for r in requests], S, rounds=3, budget=1)
                for seed in (7, 8):
                    rows = frame[(frame.S == S) & (frame.seed == seed) & (frame.policy == 'drop_backfill')]
                    arrivals = rows[rows.kind == 'request'].sort_values('step')
                    self.assertEqual(arrivals.shards_included.tolist(), [len(x) for x in expected])
                    empty = arrivals[arrivals.shards_included == 0]
                    self.assertTrue(empty.unavailable.all())
                    self.assertTrue(empty[['cal_macro_f1', 'cal_fpr', 'cal_recall_attack',
                                           'client_f1_min', 'client_f1_mean']].isna().all().all())
                    final = rows[rows.kind == 'backlog_done'].iloc[0]
                    self.assertEqual(final.shards_included, S)
                    self.assertEqual(final.pending, 0)
        spread = self.steps['spread']
        self.assertGreater(len(spread[(spread.policy == 'drop_backfill') & spread.unavailable]), 0)

    def test_single_references_are_paired_to_their_own_stream(self):
        for family in FAMILIES:
            frame = self.steps[family]
            refs = frame[frame.policy == 'full_retrain_single']
            self.assertEqual(len(refs), 2*2*4)
            self.assertTrue((refs.model_shards == 1).all())
            for S in (4, 5):
                requests = self.streams[family]['streams'][str(S)]
                for seed in (7, 8):
                    rows = refs[(refs.S == S) & (refs.seed == seed)].sort_values('step')
                    self.assertEqual(rows.step.tolist(), [5, 10, 15, 20])
                    for row in rows.itertuples():
                        self.assertEqual(row.forget_request_hash, requests[row.step - 1]['forget_request_hash'])
                        self.assertEqual(row.delta_rows_x_rounds, 3*(800 - sum(r['rows'] for r in requests[:row.step])))


if __name__ == '__main__':
    unittest.main()
