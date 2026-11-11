"""src/fu/shard_exact.py: the callable sharded-unlearning component and its cost records.

EquivalenceTests runs the unmodified, hash-locked v2 stream runner (scripts/shard/run_shard_stream_v2.py) on synthetic data,
by replacing shard_v2.start_run (the only place that loads and hash-checks the real data), and requires the component to
reproduce every step of it: event, round counts, cumulative cost and the calibration metrics, exactly.
"""
import contextlib
import io
import json
import runpy
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
import pandas as pd
import torch
from scipy.special import expit

import scripts.shard.shard_v2 as v2
from scripts.shard import replay_v2_stream, shard_service
from src.data.federated import partition
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_cost import CostMeter, TRAIN_PHASES
from src.fu.shard_exact import (EXACTNESS_SCOPE, POLICIES, ForgetRequest, ShardConfig, ShardExactMethod, ShardedUnlearner,
                                assign_shards, check_design, structure_step)

ROOT = Path(__file__).resolve().parents[1]
TRAIN_CFG = {'fl': {'rounds': 4, 'local_epochs': 1, 'batch_size': 128, 'lr': 1e-2},
             'unlearn_budgets': [1, 4], 'shard_fixed_rounds': 2, 'backfill_rounds': 3}


def synthetic(rows, dim=6, seed=0):
    """Two noisy Gaussian classes; splits disjoint by construction."""
    rng = np.random.default_rng(seed); direction = rng.normal(size=dim)

    def split(n, offset):
        y = (rng.random(n) < .7).astype(np.int64)
        X = (rng.normal(size=(n, dim)) + 1.5*(y[:, None] - .5)*direction).astype(np.float32)
        return {'X': X, 'y': y, 'row_id': np.arange(offset, offset + n)}
    return {'train': split(rows, 0), 'validation': split(rows//4, 10**6), 'calibration': split(rows//4, 2*10**6)}


def config(**overrides):
    return ShardConfig.from_train_config(TRAIN_CFG, **overrides)


def weights_equal(a, b):
    return all(torch.equal(x, y) for x, y in zip(a.state_dict().values(), b.state_dict().values()))


class TorchCase(unittest.TestCase):
    def setUp(self):
        threads, deterministic = torch.get_num_threads(), torch.are_deterministic_algorithms_enabled()
        self.addCleanup(torch.set_num_threads, threads)
        self.addCleanup(torch.use_deterministic_algorithms, deterministic)
        torch.set_num_threads(2); torch.use_deterministic_algorithms(True)


class StructureTests(unittest.TestCase):
    def test_assign_and_structure_match_the_v2_protocol(self):
        for shards in (1, 2, 4, 5, 10):
            self.assertEqual(assign_shards(shards, 20), v2.assign(shards, 20))
        rng = np.random.default_rng(3)
        for trial in range(40):
            shards, merge_below = int(rng.choice([2, 4, 5, 10])), int(rng.choice([0, 3]))
            assignment = v2.assign(shards, 20, seed=trial); alive = list(range(20)); stream = []
            for step in range(12):
                client = int(rng.choice(alive)); kind = 'client' if rng.random() < .6 else 'sample'
                if kind == 'client': alive.remove(client)
                stream.append({'step': step + 1, 'client': client, 'kind': kind})
            current = assignment
            for request, expected in zip(stream, v2.structure_plan(assignment, stream, merge_below)):
                got = structure_step(current, request['client'], request['kind'], merge_below)
                self.assertEqual(got, {k: expected[k] for k in got})
                current = got['assign_after']

    def test_design_and_request_kind_are_checked(self):
        with self.assertRaises(ValueError): check_design([[0, 1, 2, 3], [4, 5, 6]], 4)
        with self.assertRaises(ValueError): structure_step([[0, 1], [2, 3]], 9, 'client', 0)
        with self.assertRaises(ValueError): structure_step([[0, 1], [2, 3]], 0, 'class', 0)


class ComponentTests(TorchCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic(1600, seed=1)
        cls.parts = partition(cls.data['train']['y'], 3, 5., 8)
        cls.assignment = assign_shards(2, 8)

    def unlearner(self, policy='shard_full', **kw):
        d = self.data
        return ShardedUnlearner(d['train'], d['validation'], self.parts, self.assignment, 11, policy, config(**kw.pop('cfg', {})), **kw)

    def test_fit_equals_plain_fedavg_and_metering_does_not_change_the_model(self):
        from src.fl.fedavg_core import ClientWorker, fedavg, model
        fed = self.unlearner(); fed.fit(); bare = self.unlearner(measure_payload=False); bare.fit()
        tr = self.data['train']
        for j, clients in enumerate(self.assignment):
            workers = [ClientWorker(c, tr['X'][self.parts[c]], tr['y'][self.parts[c]], tr['row_id'][self.parts[c]], 'cpu') for c in clients]
            m = model(tr['X'].shape[1], [64, 32], 11 + 1000*j, 'cpu')
            for r in range(TRAIN_CFG['fl']['rounds']):                      # the round seeds of the v2 Trainer
                m, _ = fedavg(m, workers, TRAIN_CFG['fl'], 1, 11 + 1000*j + r*10000)
            self.assertTrue(weights_equal(m, fed.shards[j].model))
            self.assertTrue(weights_equal(m, bare.shards[j].model))
        self.assertEqual(fed.verify_bit_exact(), {'bit_exact': True, 'shards_checked': 2, 'per_shard': {0: True, 1: True},
                                                  'included_equals_active': True, 'exact': True, 'structure_ok': True})

    def test_every_policy_is_exact_after_a_stream(self):
        base = self.unlearner(); base.fit()
        target = self.assignment[0][0]; other = self.assignment[1][0]
        for policy in POLICIES:
            fed = base.fork(policy)
            held = self.parts[target]
            results = [fed.unlearn(ForgetRequest(target, held[::3])), fed.unlearn(ForgetRequest(other)),
                       fed.unlearn(ForgetRequest(target, held[1::3]))]
            if policy == 'drop_backfill': results.append(fed.finish_backlog())
            self.assertEqual([r.kind for r in results[:3]], ['sample', 'client', 'sample'], policy)
            for r in results:
                self.assertTrue(r.exact and r.structure_ok, (policy, r.step))
                self.assertEqual(r.exactness_scope, EXACTNESS_SCOPE[policy])
            check = fed.verify_bit_exact()
            self.assertTrue(check['bit_exact'] and check['included_equals_active'], (policy, check))
            self.assertEqual(int(fed.alive.sum()), len(fed.alive) - len(held[::3]) - len(self.parts[other]) - len(held[1::3]))
            self.assertNotIn(other, fed.present)
        self.assertEqual(base.step, 0)                                       # forks never touch the base
        self.assertTrue(base.alive.all())

    def test_round_count_scope_is_reported(self):
        base = self.unlearner(); base.fit()
        codes = lambda r: {w['code'] for w in r.structured_warnings}
        fixed = base.fork('shard_full').unlearn(ForgetRequest(0, self.parts[0][:5]))
        adaptive = base.fork('utility_stop').unlearn(ForgetRequest(0, self.parts[0][:5]))
        self.assertNotIn('ROUND_COUNT_DEPENDS_ON_PRE_DELETION_STATE', codes(fixed))
        self.assertIn('ROUND_COUNT_DEPENDS_ON_PRE_DELETION_STATE', codes(adaptive))
        self.assertEqual(fixed.cost['rounds'], TRAIN_CFG['fl']['rounds'])
        self.assertLessEqual(adaptive.details['phase1_rounds'], TRAIN_CFG['fl']['rounds'])

    def test_invalid_requests_fail_before_any_change(self):
        fed = self.unlearner()
        with self.assertRaisesRegex(RuntimeError, 'fit'): fed.unlearn(ForgetRequest(0))
        fed.fit(); held = self.parts[0]
        for bad in (ForgetRequest(99), ForgetRequest(0, []), ForgetRequest(0, [held[0], held[0]]), ForgetRequest(0, self.parts[1][:2])):
            with self.assertRaises(ValueError): fed.unlearn(bad)
        with self.assertRaises(ValueError): fed.unlearn(ForgetRequest(0, held[:2]), policy='nope')
        self.assertTrue(fed.alive.all()); self.assertEqual(fed.step, 0)
        fed.unlearn(ForgetRequest(0))
        with self.assertRaisesRegex(ValueError, 'holds no data'): fed.unlearn(ForgetRequest(0))
        with self.assertRaisesRegex(RuntimeError, 'already'): fed.fit()
        backlog = self.unlearner('drop_backfill', cfg={'backfill_rounds': 1}); backlog.fit()
        self.assertEqual(backlog.unlearn(ForgetRequest(1, self.parts[1][:3])).details['pending'], 1)
        with self.assertRaisesRegex(RuntimeError, 'backlog'): backlog.unlearn(ForgetRequest(2, self.parts[2][:3]), policy='shard_full')
        with self.assertRaises(ValueError):
            ShardedUnlearner(self.data['train'], self.data['validation'], self.parts, [[0, 1, 2, 3], [4, 5, 6]], 11, 'shard_full', config())
        with self.assertRaises(ValueError):
            ShardedUnlearner(self.data['train'], self.data['validation'], self.parts[:-1], self.assignment, 11, 'shard_full', config())

    def test_drop_backfill_serves_without_the_shard_until_it_is_backfilled(self):
        fed = self.unlearner('drop_backfill', cfg={'backfill_rounds': 1}); fed.fit()
        first = fed.unlearn(ForgetRequest(0, self.parts[0][:10]))
        self.assertEqual((first.shards_included, first.details['pending'], first.cost['rounds']), (1, 1, 1))
        self.assertIn('BACKLOG_PENDING', {w['code'] for w in first.structured_warnings})
        done = fed.finish_backlog()
        self.assertEqual((done.shards_included, done.cost['rounds'], done.kind), (2, TRAIN_CFG['fl']['rounds'] - 1, 'backlog_done'))
        self.assertTrue(fed.verify_bit_exact()['bit_exact'])

    def test_save_load_continues_bit_for_bit(self):
        requests = [ForgetRequest(0, self.parts[0][::4]), ForgetRequest(5), ForgetRequest(1, self.parts[1][::2]), ForgetRequest(0, self.parts[0][1::4])]
        d = self.data
        for policy, cfg in (('utility_stop', {}), ('drop_backfill', {'backfill_rounds': 1}), ('dynamicfu', {})):
            with tempfile.TemporaryDirectory() as tmp:
                live = self.unlearner(policy, cfg=cfg); live.fit(Path(tmp)/'base')
                for i, request in enumerate(requests[:2]):
                    result = live.unlearn(request, checkpoint_dir=Path(tmp)/f'step{i}')
                self.assertEqual(result.final_checkpoint, (Path(tmp)/'step1').as_posix())
                self.assertGreater(result.cost['checkpoint_bytes_on_disk']['models'], 0)
                restored = ShardedUnlearner.load(Path(tmp)/'step1', d['train'], d['validation'], self.parts)
                self.assertEqual((restored.step, restored.present, restored.assignment, restored.reference, restored.totals),
                                 (live.step, live.present, live.assignment, live.reference, live.totals))
                self.assertEqual(len(restored.pending), len(live.pending))
                for request in requests[2:]:
                    a, b = live.unlearn(request), restored.unlearn(request)
                    self.assertEqual((a.details, a.cost['rounds'], a.cost['cumulative_unlearning'], a.event),
                                     (b.details, b.cost['rounds'], b.cost['cumulative_unlearning'], b.event), policy)
                self.assertEqual(live.included(), restored.included())
                for j in live.included():
                    self.assertTrue(weights_equal(live.shards[j].model, restored.shards[j].model), (policy, j))
                self.assertTrue(np.array_equal(live.margin(d['calibration']['X']), restored.margin(d['calibration']['X'])))
                state = json.loads((Path(tmp)/'step1'/'state.json').read_text())
                self.assertEqual([h['step'] for h in state['history']], [1, 2])
                self.assertTrue(all(len(h['forget_request_hash']) == 64 for h in state['history']))

    def test_load_rejects_other_data_and_tampered_files(self):
        d = self.data
        with tempfile.TemporaryDirectory() as tmp:
            fed = self.unlearner(); fed.fit(Path(tmp)/'base')
            with self.assertRaises(FileExistsError): fed.save(Path(tmp)/'base')
            other = dict(d['train'], X=d['train']['X'] + 1)
            with self.assertRaisesRegex(ValueError, "different data.*'train'"):
                ShardedUnlearner.load(Path(tmp)/'base', other, d['validation'], self.parts)
            moved = [p.copy() for p in self.parts]; moved[0], moved[1] = moved[1], moved[0]
            with self.assertRaisesRegex(ValueError, "different data.*'partition'"):
                ShardedUnlearner.load(Path(tmp)/'base', d['train'], d['validation'], moved)
            with open(Path(tmp)/'base'/'shard_0.pt', 'ab') as handle: handle.write(b'x')
            with self.assertRaisesRegex(ValueError, 'hash mismatch'):
                ShardedUnlearner.load(Path(tmp)/'base', d['train'], d['validation'], self.parts)

    def test_method_interface_runs_from_a_checkpoint_into_a_new_directory(self):
        d = self.data
        with tempfile.TemporaryDirectory() as tmp:
            self.unlearner().fit(Path(tmp)/'base')
            method = ShardExactMethod(d['train'], d['validation'], self.parts)
            result = method.run(Path(tmp)/'base', 3, self.parts[3][:7], {'policy': 'shard_fixed'}, Path(tmp)/'after')
            self.assertEqual((result.policy, result.rows, result.cost['rounds']), ('shard_fixed', 7, TRAIN_CFG['shard_fixed_rounds']))
            self.assertTrue((Path(tmp)/'after'/'state.json').exists())
            with self.assertRaises(FileExistsError): method.run(Path(tmp)/'base', 3, None, None, Path(tmp)/'after')
            json.dumps(result.to_json())                                  # serialisable as is
            self.assertLessEqual({'compute_seconds_sum', 'makespan_seconds_simulated', 'wall_seconds_total'}, set(result.timing_breakdown))

    def test_five_clients_in_one_shard(self):
        d = synthetic(600, seed=2); parts = partition(d['train']['y'], 1, 5., 5)
        fed = ShardedUnlearner(d['train'], d['validation'], parts, assign_shards(1, 5), 5, 'shard_full', config()); fed.fit()
        result = fed.unlearn(ForgetRequest(2, parts[2][:20]))
        self.assertEqual((result.retrained_shard, result.cost['clients_selected']), (0, [0, 1, 2, 3, 4]))
        self.assertTrue(fed.verify_bit_exact()['bit_exact'])


class CostTests(TorchCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic(1600, seed=1)
        cls.parts = partition(cls.data['train']['y'], 3, 5., 8)
        cls.assignment = assign_shards(2, 8)

    def fitted(self, policy):
        d = self.data
        fed = ShardedUnlearner(d['train'], d['validation'], self.parts, self.assignment, 11, policy, config())
        return fed, fed.fit()

    def test_per_client_records_add_up_to_the_summary(self):
        fed, base = self.fitted('shard_full')
        rounds = TRAIN_CFG['fl']['rounds']
        self.assertEqual((base['cost']['rounds'], len(base['client_costs'])), (2*rounds, 8*rounds))
        self.assertEqual(base['cost']['rows_x_rounds'], len(fed.alive)*rounds)
        self.assertEqual(fed.totals, {'rows_x_rounds': 0, 'rounds': 0, 'train_bytes_raw': 0})          # the base fit is not unlearning cost
        self.assertLess(base['cost']['makespan_parallel_shards_seconds_simulated'], base['cost']['makespan_seconds_simulated'])
        target = self.assignment[1][0]; shard_clients = self.assignment[1]
        result = fed.unlearn(ForgetRequest(target, self.parts[target][:9]))
        cost, rows = result.cost, result.client_costs
        retained = sum(len(self.parts[c]) for c in shard_clients) - 9
        self.assertEqual(cost['clients_selected'], sorted(shard_clients))
        self.assertEqual((cost['rounds'], cost['client_rounds'], cost['rows_x_rounds']), (rounds, rounds*len(shard_clients), rounds*retained))
        self.assertEqual({r['phase'] for r in rows}, {'retrain'})
        self.assertEqual(sum(r['samples'] for r in rows), cost['rows_x_rounds'])
        self.assertEqual(cost['train_bytes_raw'], 2*fed.model_bytes*cost['client_rounds'])              # the byte formula of the v2 runners
        self.assertEqual((cost['bytes_up_raw'], cost['bytes_down_raw']), (fed.model_bytes*cost['client_rounds'],)*2)
        self.assertEqual(cost['bytes_up_packed'], sum(r['bytes_up_packed'] for r in rows))
        self.assertTrue(all(0 < r['bytes_up_packed'] < 2*fed.model_bytes and r['bytes_down_packed'] > 0 for r in rows))
        self.assertEqual(sum(result.client_bytes_up.values()), cost['bytes_up_packed'])
        self.assertEqual(sorted(result.client_bytes_down), sorted(str(c) for c in shard_clients))
        self.assertAlmostEqual(cost['compute_seconds_sum'], sum(r['compute_seconds'] for r in rows))
        self.assertLessEqual(cost['makespan_seconds_simulated'], cost['compute_seconds_sum'] + cost['server_aggregate_seconds'] + cost['policy_eval_seconds'] + 1e-9)
        self.assertGreater(cost['makespan_seconds_simulated'], max(r['compute_seconds'] for r in rows))
        self.assertGreaterEqual(cost['wall_seconds_total'], cost['compute_seconds_sum'])
        self.assertEqual(cost['cumulative_unlearning'], {'rows_x_rounds': cost['rows_x_rounds'], 'rounds': rounds, 'train_bytes_raw': cost['train_bytes_raw']})
        self.assertEqual((cost['models_stored'], cost['inference_forward_passes'], cost['state_bytes_in_memory']), (2, 2, 2*fed.model_bytes))
        self.assertEqual(len(result.local_losses), len(rows))
        if cost['rss_bytes_end'] is not None:
            self.assertGreaterEqual(cost['rss_bytes_peak_sampled'], cost['rss_bytes_start'])
        else:
            self.assertIn('MEMORY_UNAVAILABLE', {w['code'] for w in result.structured_warnings})

    def test_policy_and_probe_work_is_separated_from_training(self):
        fed, _ = self.fitted('dynamicfu')
        target = self.assignment[0][1]
        result = fed.unlearn(ForgetRequest(target, self.parts[target][:9]))
        probes = [r for r in result.client_costs if r['phase'] == 'contribution']
        self.assertEqual(sorted(r['client_id'] for r in probes), list(range(8)))                        # every present client is probed
        self.assertEqual(result.cost['client_rounds'], sum(r['phase'] in TRAIN_PHASES for r in result.client_costs))
        self.assertGreater(result.cost['bytes_up_raw'], result.cost['train_bytes_raw']//2)              # probe traffic is counted
        self.assertGreater(result.cost['policy_eval_seconds'], 0)
        stop, _ = self.fitted('utility_stop')
        self.assertGreater(stop.unlearn(ForgetRequest(target, self.parts[target][:9])).cost['policy_eval_seconds'], 0)
        full, _ = self.fitted('shard_full')
        self.assertEqual(full.unlearn(ForgetRequest(target, self.parts[target][:9])).cost['policy_eval_seconds'], 0)

    def test_backend_warm_up_runs_once_per_process_and_is_reported(self):
        import src.fu.shard_exact as component
        d = self.data
        with mock.patch.object(component, '_WARM', False):
            cold = ShardedUnlearner(d['train'], d['validation'], self.parts, self.assignment, 11, 'shard_fixed', config())
            fit = cold.fit(); base = list(cold.shards)
            self.assertGreater(fit['cost']['backend_warmup_seconds'], 0)
            self.assertEqual(component.warm_up(cold.dim, cold.config), 0.)                                 # already warm
            result = cold.unlearn(ForgetRequest(0, self.parts[0][:4]))
            self.assertEqual(result.cost['backend_warmup_seconds'], 0.)
            self.assertIn('backend_warmup_seconds', result.timing_breakdown)
        plain, _ = self.fitted('shard_fixed')                                    # the same fit without a warm-up before it
        self.assertTrue(all(weights_equal(a.model, b.model) for a, b in zip(base, plain.shards)))
        self.assertTrue(cold.verify_bit_exact()['bit_exact'])

    def test_unmetered_payload_and_state_helpers(self):
        d = self.data
        fed = ShardedUnlearner(d['train'], d['validation'], self.parts, self.assignment, 11, 'shard_fixed', config(), measure_payload=False)
        fed.fit()
        result = fed.unlearn(ForgetRequest(0, self.parts[0][:4]))
        self.assertIsNone(result.cost['bytes_up_packed'])
        self.assertEqual(sum(result.client_bytes_up.values()), result.cost['bytes_up_raw'])             # falls back to raw bytes
        clients = fed.client_state()
        self.assertEqual([c['client_id'] for c in clients], list(range(8)))
        self.assertEqual(clients[0]['samples_retained'], len(self.parts[0]) - 4)
        self.assertEqual(sum(clients[0]['class_counts']), clients[0]['samples_retained'])
        bench = fed.benchmark_inference(d['calibration']['X'], repeats=2)
        self.assertEqual((bench['forward_passes'], bench['rows']), (2, len(d['calibration']['y'])))
        self.assertIsInstance(CostMeter().summary()['makespan_method'], str)


class ServiceTests(TorchCase):
    """scripts/shard/shard_service.py: state directories, request files and the command line."""

    def test_init_unlearn_verify(self):
        data = synthetic(1600, seed=1)
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp); layout = dict(clients=8, alpha=5., partition_seed=3, train_config='repo')
            fed, fit = shard_service.init_state(data, tmp/'s0', 2, 11, 'shard_fixed', **layout)
            self.assertEqual({p.name for p in (tmp/'s0').iterdir()}, {'state.json', 'rows.npz', 'shard_0.pt', 'shard_1.pt', 'layout.json',
                                                                    'result.json', 'round_costs.csv', 'client_costs.csv'})
            self.assertEqual(len(pd.read_csv(tmp/'s0'/'client_costs.csv')), 8*15)
            held = fed.parts[2]; ids = data['train']['row_id'][held[:6]].tolist()
            (tmp/'r1.json').write_text(json.dumps({'client': 2, 'forget_row_ids': ids}))
            (tmp/'r2.json').write_text(json.dumps({'client': 5}))
            request = shard_service.read_request(data, tmp/'r1.json')
            self.assertTrue(np.array_equal(request.rows, held[:6]))
            for bad in ({'client': 2, 'forget_row_ids': [10**9]}, {'client': 2, 'forget_row_ids': []}, {'forget_row_ids': ids},
                        {'client': 2, 'kind': 'sample'}, {'client': 2, 'rows': ids}):
                (tmp/'bad.json').write_text(json.dumps(bad))
                with self.assertRaises(ValueError): shard_service.read_request(data, tmp/'bad.json')
            with self.assertRaises(FileExistsError): shard_service.init_state(data, tmp/'s0', 2, 11, 'shard_fixed', **layout)
            fed1, first = shard_service.apply_request(data, tmp/'s0', request, tmp/'s1')
            self.assertEqual((first.kind, first.rows, first.policy, first.cost['rounds']), ('sample', 6, 'shard_fixed', 7))
            summary = json.loads((tmp/'s1'/'result.json').read_text())
            self.assertEqual((summary['operation'], summary['step'], summary['final_checkpoint']), ('unlearn', 1, (tmp/'s1').as_posix()))
            self.assertEqual(summary['clients'][2]['samples_retained'], len(held) - 6)
            self.assertNotIn('client_costs', summary)
            self.assertEqual(len(pd.read_csv(tmp/'s1'/'client_costs.csv')), 7*len(fed.assignment[first.retrained_shard]))
            with mock.patch('src.data.loading.load_data', lambda path: data), contextlib.redirect_stdout(io.StringIO()) as out:
                shard_service.main(['unlearn', '--original-run', str(tmp/'s1'), '--forget-request', str(tmp/'r2.json'),
                                    '--method', 'shard_full', '--output', str(tmp/'s2')])
                shard_service.main(['verify', '--original-run', str(tmp/'s2')])
            printed = [json.loads(line) for line in out.getvalue().splitlines()]
            self.assertEqual((printed[0]['step'], printed[0]['kind'], printed[0]['policy'], printed[0]['rounds']), (2, 'client', 'shard_full', 15))
            self.assertTrue(printed[1]['bit_exact'] and printed[1]['included_equals_active'])
            self.assertTrue(printed[0]['exact'] and printed[0]['structure_ok'])
            with open(tmp/'s2'/'shard_0.pt', 'ab') as handle: handle.write(b'x')
            with mock.patch('src.data.loading.load_data', lambda path: data), self.assertRaisesRegex(ValueError, 'hash mismatch'):
                shard_service.main(['verify', '--original-run', str(tmp/'s2')])


class EquivalenceTests(TorchCase):
    """The component against the unmodified v2 stream runner, on synthetic data."""

    def run_v2(self, data, design):
        with tempfile.TemporaryDirectory() as tmp, \
                mock.patch.object(v2, 'start_run', lambda out, runner: ('synthetic', design, TRAIN_CFG, data)), \
                mock.patch.object(sys, 'argv', ['run_shard_stream_v2.py', str(Path(tmp)/'out')]), \
                contextlib.redirect_stdout(io.StringIO()):
            return runpy.run_path(str(ROOT/'scripts'/'shard'/'run_shard_stream_v2.py'))

    def check(self, design, expect_events):
        data = synthetic(4000, seed=4)          # a size and seed for which the runner's 20-client partition (seed 42) is valid
        ref = self.run_v2(data, design)
        tr, va, ca, stream, parts = data['train'], data['validation'], data['calibration'], ref['stream'], ref['parts']
        want = {(r['S'], r['seed'], r['policy'], r['step']): r for r in ref['rows']}
        ends = {(c['S'], c['seed'], c['policy']): c for c in ref['checks']}
        cfg = config(merge_below=design['merge_below'], min_clients_design=design['min_clients_design'])
        seen, metrics = set(), set()
        for shards in design['shards_run2']:
            for seed in design['seeds']:
                base = ShardedUnlearner(tr, va, parts, assign_shards(shards, 20), seed, 'shard_full', cfg, measure_payload=False)
                base.fit()
                for policy in POLICIES:
                    fed = base.fork(policy)
                    steps = [lambda r=r: fed.unlearn(ForgetRequest(r['client'], r['f'])) for r in stream]
                    if policy == 'drop_backfill': steps.append(fed.finish_backlog)
                    for step, apply in enumerate(steps, 1):
                        result = apply(); row = want[shards, seed, policy, step]; label = (shards, seed, policy, step)
                        got = detector_metrics(ca['y'], expit(fed.margin(ca['X'])))     # the split and metrics the runner reports
                        self.assertEqual((got['macro_f1'], got['fpr'], got['recall_attack']),
                                         (row['cal_macro_f1'], row['cal_fpr'], row['cal_recall_attack']), label)
                        metrics.add(got['macro_f1'])
                        if step <= len(stream):
                            self.assertEqual((result.kind, result.event), (row['kind'], row['event']), label)
                            seen.add(result.event)
                        total = result.cost['cumulative_unlearning']
                        self.assertEqual((result.exact, result.structure_ok, result.shards_included), (row['exact'], row['structure_ok'], row['shards_included']), label)
                        self.assertEqual((total['rows_x_rounds'], total['rounds'], total['train_bytes_raw']), (row['rows_x_rounds'], row['rounds'], row['bytes']), label)
                        for key, value in result.details.items():
                            if step <= len(stream):                       # the runner's backlog row carries no policy outputs
                                self.assertEqual(value, row[key], (label, key))
                    check = fed.verify_bit_exact(); end = ends[shards, seed, policy]
                    self.assertEqual((check['bit_exact'], check['included_equals_active'], check['structure_ok'], check['shards_checked']),
                                     (end['bit_exact'], end['included_equals_active'], end['structure_ok_end'], end['shards_checked']), (shards, seed, policy))
                    self.assertTrue(check['bit_exact'])
                # the shard models themselves, against the runner's own Trainer
                fresh = ref['Trainer'](np.sort(np.concatenate([parts[c] for c in base.assignment_initial[0]])), seed)
                for _ in range(TRAIN_CFG['fl']['rounds']): fresh.step()
                self.assertTrue(weights_equal(fresh.m, base.shards[0].model))
        self.assertTrue(expect_events <= seen, seen)
        self.assertGreater(len(metrics), 20, 'the metric comparison must not be degenerate')
        return ref, data, cfg

    def test_stream_with_merging(self):
        ref, data, cfg = self.check({'train_config': 'smoke', 'seeds': [7], 'shards_run1': [1], 'shards_run2': [4, 5],
                                     'min_clients_design': 4, 'merge_below': 3}, {'retrain', 'merge_4_into_1'})
        rows = [r for r in ref['rows'] if r['policy'] == 'utility_stop']
        self.assertTrue(any(r['fallback_rounds'] > 0 for r in rows), 'the synthetic stream must exercise the fallback')
        self.assertTrue(any(r['pending'] > 0 for r in ref['rows'] if r['policy'] == 'drop_backfill' and r['step'] <= 10))
        self.assertLessEqual({'ctrl', 'drop'}, {r['stop_reason'] for r in ref['rows'] if r['policy'] == 'dynamicfu'})

        # the replay script: same request stream, and its comparison against a steps.csv written by the runner
        mine = replay_v2_stream.request_stream(data['train']['y'], ref['parts'])
        self.assertEqual([(r['step'], r['client'], r['kind'], r['rows']) for r in mine], [(r['step'], r['client'], r['kind'], r['rows']) for r in ref['stream']])
        self.assertTrue(all(np.array_equal(a['f'], b['f']) for a, b in zip(mine, ref['stream'])))
        with tempfile.TemporaryDirectory() as tmp:
            pd.DataFrame(ref['rows']).to_csv(Path(tmp)/'steps.csv', index=False)
            stored = replay_v2_stream.read_steps(Path(tmp)/'steps.csv')
        chosen = ['utility_stop', 'dynamicfu', 'drop_backfill']                # one fit, then every policy from the same base
        steps, fit, results, ends = replay_v2_stream.replay(data, 5, 7, chosen, cfg)
        outcome = replay_v2_stream.compare(steps, stored)
        self.assertTrue(outcome['pass'] and outcome['metrics_bit_equal'], outcome['mismatches'])
        self.assertEqual({p: v['steps_compared'] for p, v in outcome['policies'].items()}, {'utility_stop': 10, 'dynamicfu': 10, 'drop_backfill': 11})
        self.assertEqual([len(results[p]) for p in chosen], [10, 10, 11])
        self.assertTrue(all(ends[p]['bit_exact'] and ends[p]['included_equals_active'] for p in chosen))
        self.assertTrue(fit['reproducible']['bit_exact'])                         # the base every policy starts from
        self.assertEqual(fit['cost']['rounds'], 5*TRAIN_CFG['fl']['rounds'])
        self.assertEqual(fit['inference']['forward_passes'], 5)
        wrong = stored.copy(); wrong.loc[(wrong.policy == 'drop_backfill') & (wrong.step == 4), 'rounds'] += 1
        failed = replay_v2_stream.compare(steps, wrong)
        self.assertEqual([(m['policy'], m['step'], m['field']) for m in failed['mismatches']], [('drop_backfill', 4, 'rounds')])
        self.assertEqual((failed['policies']['drop_backfill']['first_mismatch_step'], failed['policies']['dynamicfu']['first_mismatch_step']), (4, None))
        drift = stored.copy(); drift['cal_fpr'] += 1e-9
        self.assertFalse(replay_v2_stream.compare(steps, drift)['pass'])
        self.assertTrue(replay_v2_stream.compare(steps, drift, tolerance=1e-8)['pass'])
        self.assertFalse(replay_v2_stream.compare(steps, stored[stored.step != 3])['pass'])

    def test_stream_with_empty_shards(self):
        self.check({'train_config': 'smoke', 'seeds': [8], 'shards_run1': [1], 'shards_run2': [10],
                    'min_clients_design': 2, 'merge_below': 0}, {'retrain', 'removed_empty'})


if __name__ == '__main__':
    unittest.main()
