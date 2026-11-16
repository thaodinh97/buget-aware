"""Protocol evaluation checks use invented metrics, never experimental evidence."""
import copy
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np
import pandas as pd

from scripts.shard import evaluate_shard_v3 as ev
from scripts.shard.shard_v2 import DATA_SHA256, SMOKE_CONFIG, assign
from scripts.shard.shard_v3 import (DESIGN, DESIGNS, EXPECTED_CLIENT_ROWS, RUNNER,
                                   machine_info, source_hashes)
from scripts.shard.stream_v3 import FAMILIES, STREAM, backfill_plan, batch_windows, plan_stream
from scripts.shard.train_config import CONFIGS
from src.fu.shard_exact import ShardConfig


def artifacts():
    """A complete smoke-shaped artifact, with equal utility for every available model."""
    seeds, shards = [7, 8], [4, 5]
    n, cfg = EXPECTED_CLIENT_ROWS, SMOKE_CONFIG
    rb = cfg['fl']['rounds']
    streams, records, rows, exactness = {}, {}, [], []
    for family in FAMILIES:
        streams[family] = {'format': 1, 'family': family, 'client_rows': n, 'stream_config': STREAM,
                           'assignments': {}, 'streams': {}}
        records[family] = {'design_name': 'smoke_v3', 'design': DESIGNS['smoke_v3'],
                           'train_config_name': 'smoke', 'train_config': cfg, 'stream_config': STREAM, 'family': family}
        for S in shards:
            A = assign(S)
            st = [{**r, 'forget_request_hash': str(r['step']).zfill(64)} for r in plan_stream(family, n, A)]
            streams[family]['assignments'][str(S)] = A
            streams[family]['streams'][str(S)] = st
            backfill = backfill_plan([r['shard'] for r in st], S, rb, cfg['backfill_rounds'])
            for seed in seeds:
                for policy in ev.POLICIES:
                    W = ev.BATCHES.get(policy, 0)
                    processing = batch_windows([r['time'] for r in st], W)
                    groups = {}
                    for request, at in zip(st, processing):
                        groups.setdefault(at, []).append(request)
                    cost, rounds = 0, 0
                    remaining = np.asarray(n).copy()
                    for at, batch in groups.items():
                        pending = 0
                        for request in batch:
                            remaining[request['client']] -= request['rows']
                            pending += request['rows']
                            included = backfill[request['step']-1] if policy == 'drop_backfill' else list(range(S))
                            row = {'family': family, 'S': S, 'model_shards': S, 'seed': seed, 'policy': policy,
                                   **request, 'kind': 'request', 'processed_at': at, 'latency': at-request['time'],
                                   'exposure': request['rows']*(at-request['time']), 'pending_rows': pending if W else 0,
                                   'exact': np.nan if W else policy != 'none', 'structure_ok': np.nan if W else policy != 'none',
                                   'exactness_scope': ev.SCOPES.get(policy, 'none'), 'shards_included': len(included),
                                   'included_shards': json.dumps(included), 'unavailable': not included,
                                   'cal_macro_f1': .8 if included else np.nan, 'cal_fpr': .1 if included else np.nan,
                                   'cal_recall_attack': .8 if included else np.nan, 'rows_x_rounds': cost, 'rounds': rounds,
                                   'delta_rows_x_rounds': 0, 'delta_rounds': 0, 'bytes': 0, 'seconds': 0.,
                                   'fallback_rounds': 0, 'condition_met': True}
                            if policy == 'shard_full':
                                delta = int(remaining[A[request['shard']]].sum())*rb
                                cost += delta
                                rounds += rb
                                row.update(rows_x_rounds=cost, delta_rows_x_rounds=delta, rounds=rounds, delta_rounds=rb)
                            rows.append(row)
                        if W:
                            touched = sorted({r['shard'] for r in batch})
                            delta = sum(int(remaining[A[j]].sum()) for j in touched)*rb
                            cost += delta
                            rounds += len(touched)*rb
                            rows.append({**rows[-1], 'kind': 'batch_end', 'time': at, 'rows': pending, 'pending_rows': 0,
                                         'exact': True, 'structure_ok': True, 'rows_x_rounds': cost, 'rounds': rounds,
                                         'delta_rows_x_rounds': delta, 'delta_rounds': len(touched)*rb,
                                         'exposure': sum(r['rows']*(at-r['time']) for r in batch)})
                    if policy == 'drop_backfill':
                        rows.append({**rows[-1], 'kind': 'backlog_done', 'step': 21, 'shards_included': S,
                                     'included_shards': json.dumps(list(range(S))), 'unavailable': False,
                                     'cal_macro_f1': .8, 'cal_fpr': .1, 'cal_recall_attack': .8})
                    if policy != 'none':
                        exactness.append({'family': family, 'S': S, 'seed': seed, 'policy': policy, 'bit_exact': True,
                                          'included_equals_active': True, 'structure_ok_end': True, 'exact': True,
                                          'shards_checked': S, 'exactness_scope': ev.SCOPES[policy],
                                          'batch_matches_shard_full': True if W else np.nan})
                for step in (5, 10, 15, 20):
                    rows.append({**rows[-1], 'policy': 'full_retrain_single', 'kind': 'reference', 'step': step,
                                 'time': st[step-1]['time'], 'model_shards': 1, 'shards_included': 1, 'unavailable': False,
                                 'cal_macro_f1': .8, 'cal_fpr': .1, 'cal_recall_attack': .8})
    return pd.DataFrame(rows), pd.DataFrame(exactness), streams, records


def standard_records():
    records = {}
    for family in FAMILIES:
        record = {'design_name': 'v3', 'design': DESIGN, 'train_config_name': 'standard',
                  'train_config': CONFIGS['standard'], 'family': family, 'runner': RUNNER,
                  'data_content_sha256': DATA_SHA256, 'client_rows': EXPECTED_CLIENT_ROWS,
                  'stream_config': STREAM, 'sha256_at_start': source_hashes(), 'machine': {**machine_info(), 'threads': 1},
                  'component_config': ShardConfig.from_train_config(CONFIGS['standard']).to_json(),
                  'completed': True, 'reproducible': True, 'attempt': 1,
                  'reproducibility_checks': [{'S': S, 'seed': seed, 'reproducible': True}
                                              for S in DESIGN['shards'] for seed in DESIGN['seeds']]}
        records[family] = {**record, 'attempts': [copy.deepcopy(record)], 'selected_attempt': 1}
    return records


class EvaluationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.steps, cls.exactness, cls.streams, cls.records = artifacts()

    def coverage(self, steps=None, exactness=None):
        return ev.coverage(self.steps if steps is None else steps, self.exactness if exactness is None else exactness,
                           self.streams, [7, 8], [4, 5])

    def test_complete_smoke_exercises_all_gates_but_cannot_confirm(self):
        result = ev.evaluate_frames(self.steps, self.exactness, self.streams, self.records)
        self.assertTrue(result['coverage']['pass'])
        for gate in ('G1', 'G2', 'G3', 'G4', 'G6'):
            self.assertTrue(result['gates'][gate]['pass'], (gate, result['gates'][gate]))
        self.assertFalse(result['P']['pass'])
        self.assertFalse(result['confirmed_all'])
        self.assertFalse(any(result['claims'].values()))
        json.dumps(result, allow_nan=False)

    def test_duplicate_cannot_replace_missing_row_with_same_count(self):
        damaged = pd.concat([self.steps.iloc[1:], self.steps.iloc[[1]]], ignore_index=True)
        result = self.coverage(damaged)
        self.assertFalse(result['pass'])
        self.assertEqual(result['failures']['steps']['duplicate_count'], 1)
        self.assertEqual(len(result['failures']['steps']['missing']), 1)

    def test_missing_final_check_and_nan_available_metric_fail(self):
        self.assertFalse(self.coverage(exactness=self.exactness.iloc[1:])['pass'])
        damaged = self.steps.copy()
        damaged.loc[0, 'cal_fpr'] = np.nan
        self.assertFalse(self.coverage(damaged)['pass'])

    def test_single_reference_must_be_a_single_model(self):
        damaged = self.steps.copy()
        damaged.loc[damaged.policy == 'full_retrain_single', 'model_shards'] = 4
        self.assertFalse(self.coverage(damaged)['pass'])

    def test_batch_final_equality_and_conditional_scope_are_checked(self):
        covered = self.coverage()
        damaged = self.exactness.copy()
        damaged.loc[damaged.policy == 'batch_full_2', 'batch_matches_shard_full'] = False
        self.assertFalse(ev.g1(self.steps, damaged, covered)['pass'])
        damaged = self.exactness.copy()
        damaged.loc[damaged.policy == 'utility_stop', 'exactness_scope'] = 'unconditional'
        self.assertFalse(ev.g1(self.steps, damaged, covered)['pass'])

    def test_cost_latency_and_pending_tampering_are_detected(self):
        for column, kind in [('rows_x_rounds', 'batch_end'), ('latency', 'request'), ('pending_rows', 'request')]:
            damaged = self.steps.copy()
            index = damaged[(damaged.policy == 'batch_full_2') & (damaged.kind == kind)].index[0]
            damaged.loc[index, column] += 1
            result = ev.g2(damaged, self.streams, [7, 8], [4, 5], SMOKE_CONFIG, STREAM)
            self.assertFalse(result['pass'], column)

    def test_replanned_location_and_backfill_inclusion_are_detected(self):
        streams = copy.deepcopy(self.streams)
        streams['spread']['streams']['4'][0]['client'] = 19
        self.assertFalse(ev.g2(self.steps, streams, [7, 8], [4, 5], SMOKE_CONFIG, STREAM)['pass'])
        damaged = self.steps.copy()
        index = damaged[damaged.policy == 'drop_backfill'].index[0]
        damaged.loc[index, 'included_shards'] = '[]'
        self.assertFalse(ev.g2(damaged, self.streams, [7, 8], [4, 5], SMOKE_CONFIG, STREAM)['pass'])

    def test_paired_test_rejects_duplicates_and_nonfinite_metrics(self):
        candidate = pd.DataFrame({'seed': [7, 8], 'cal_macro_f1': [.8, .8], 'cal_fpr': [.1, .1]})
        self.assertTrue(ev.compare(candidate, candidate, [7, 8])['ok'])
        self.assertFalse(ev.compare(candidate.iloc[[0, 0]], candidate, [7, 8])['ok'])
        damaged = candidate.copy()
        damaged.loc[0, 'cal_macro_f1'] = np.nan
        self.assertFalse(ev.compare(damaged, candidate, [7, 8])['ok'])

    def test_g5_uses_worst_step_per_seed_and_unavailable_is_failure(self):
        rows = []
        for family in FAMILIES:
            for seed in (7, 8):
                for step in (1, 2, 3):
                    for policy in ('drop_backfill', 'shard_full'):
                        rows.append({'family': family, 'S': 4, 'seed': seed, 'step': step, 'policy': policy,
                                     'kind': 'request', 'unavailable': False,
                                     'cal_macro_f1': .78 if policy == 'drop_backfill' and step == 2 else .8, 'cal_fpr': .1})
        frame = pd.DataFrame(rows)
        result = ev.g5(frame, [7, 8], [4], 3)
        self.assertFalse(result['pass'])
        self.assertAlmostEqual(result['comparisons'][0]['dF1_mean'], -.02)
        frame.cal_macro_f1 = .8
        self.assertTrue(ev.g5(frame, [7, 8], [4], 3)['pass'])
        frame.loc[0, 'unavailable'] = True
        result = ev.g5(frame, [7, 8], [4], 3)
        self.assertFalse(result['pass'])
        self.assertEqual(result['comparisons'][0]['unavailable_seeds'], [7])

    def test_integrity_requires_current_hashes_exact_seeds_machine_and_retry_audit(self):
        records = standard_records()
        self.assertTrue(ev.integrity(records, self.streams)['pass'])
        for mutation in ('hash', 'seed', 'machine', 'retry', 'config', 'client_rows'):
            damaged = copy.deepcopy(records)
            rec = damaged[FAMILIES[0]]
            if mutation == 'hash':
                rec['sha256_at_start']['scripts/shard/stream_v3.py'] = '0'*64
            elif mutation == 'seed':
                rec['reproducibility_checks'][0]['seed'] = 0
            elif mutation == 'machine':
                rec['machine']['torch'] = 'different'
            elif mutation == 'retry':
                rec['attempts'] = []
            elif mutation == 'config':
                rec['component_config']['hidden'] = [2, 2]
            else:
                rec['client_rows'] = [1]*20
            self.assertFalse(ev.integrity(damaged, self.streams)['pass'], mutation)

    def test_integrity_accepts_a_consistent_positive_thread_count(self):
        records = standard_records()
        for rec in records.values():
            rec['machine']['threads'] = 2
            rec['attempts'][0]['machine']['threads'] = 2
        self.assertTrue(ev.integrity(records, self.streams)['pass'])

    def test_missing_family_files_fail_closed(self):
        with tempfile.TemporaryDirectory() as directory:
            result = ev.evaluate(Path(directory))
            self.assertFalse(result['confirmed_all'])
            self.assertFalse(result['P']['pass'])
            self.assertIn('input_error', result)


if __name__ == '__main__':
    unittest.main()
