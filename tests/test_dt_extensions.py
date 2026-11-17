"""Digital Twin extensions (docs/dt_extensions.md): the shard_checkpoint policy and unlearn_batch of src/fu/shard_exact.py,
stress injection (src/dt/stress.py), the planner (src/dt/planner.py), the deadline scheduler (src/dt/scheduler.py) and the
runners in scripts/dt/."""
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np

from src.data.federated import partition
from src.dt import stress
from src.dt.stress import Condition, StressedWorker, environment, install, slow_clients
from src.fu.shard_exact import (ALL_POLICIES, EXACTNESS_SCOPE, POLICIES, ForgetRequest, ShardedUnlearner, assign_shards)
from test_shard_exact import TRAIN_CFG, TorchCase, config, synthetic, weights_equal

RB, FIXED = TRAIN_CFG['fl']['rounds'], TRAIN_CFG['shard_fixed_rounds']


class Base(TorchCase):
    @classmethod
    def setUpClass(cls):
        cls.data = synthetic(1600, seed=1)
        cls.parts = partition(cls.data['train']['y'], 3, 5., 8)
        cls.assignment = assign_shards(2, 8)

    def unlearner(self, policy='shard_full', assignment=None, **cfg):
        d = self.data
        return ShardedUnlearner(d['train'], d['validation'], self.parts, assignment or self.assignment, 11, policy, config(**cfg))


class CheckpointPolicyTests(Base):
    def test_v2_policy_list_is_unchanged(self):
        self.assertEqual(POLICIES, ('shard_full', 'shard_fixed', 'utility_stop', 'utility_stop_v1', 'dynamicfu', 'drop_backfill'))
        self.assertEqual(ALL_POLICIES[-1], 'shard_checkpoint')
        self.assertEqual(EXACTNESS_SCOPE['shard_checkpoint'], 'unconditional')

    def test_stop_equals_shard_fixed_and_continue_equals_shard_full(self):
        base = self.unlearner(); base.fit()
        request = ForgetRequest(0, self.parts[0][:10])
        fixed, full = base.fork('shard_fixed'), base.fork('shard_full')
        fixed.unlearn(request); full.unlearn(request)
        stop, go = base.fork(), base.fork()
        r_stop = stop.unlearn(request, 'shard_checkpoint', budget_seconds=0.)
        r_go = go.unlearn(request, 'shard_checkpoint', budget_seconds=1e9)
        j = r_stop.retrained_shard
        self.assertTrue(weights_equal(stop.shards[j].model, fixed.shards[j].model))
        self.assertTrue(weights_equal(go.shards[j].model, full.shards[j].model))
        self.assertEqual((stop.shards[j].rounds, go.shards[j].rounds), (FIXED, RB))
        self.assertEqual((r_stop.cost['rounds'], r_go.cost['rounds']), (FIXED, RB))
        self.assertEqual((r_stop.details['continued'], r_go.details['continued']), (False, True))
        self.assertEqual(r_stop.details['checkpoint_round'], FIXED)
        for result, fed in ((r_stop, stop), (r_go, go)):
            self.assertEqual(result.exactness_scope, 'unconditional')
            self.assertTrue(result.exact and result.structure_ok)
            self.assertNotIn('ROUND_COUNT_DEPENDS_ON_PRE_DELETION_STATE', {w['code'] for w in result.structured_warnings})
            self.assertTrue(fed.verify_bit_exact()['bit_exact'])
        self.assertIn('STOPPED_AT_CHECKPOINT', {w['code'] for w in r_stop.structured_warnings})
        self.assertNotIn('STOPPED_AT_CHECKPOINT', {w['code'] for w in r_go.structured_warnings})
        self.assertEqual(r_go.cost['policy_eval_seconds'], 0)                 # no model is scored to take the decision

    def test_decision_rule_extrapolates_the_measured_rounds(self):
        base = self.unlearner(); base.fit()
        result = base.fork().unlearn(ForgetRequest(1, self.parts[1][:5]), 'shard_checkpoint', budget_seconds=1e9, margin=.25)
        first = [r['wall_seconds'] for r in result.round_costs[:FIXED]]
        d = result.details
        self.assertAlmostEqual(d['elapsed_seconds'], sum(first))
        self.assertAlmostEqual(d['forecast_remaining_seconds'], sum(first)/FIXED*(RB - FIXED)*1.25)
        budget = d['elapsed_seconds'] + d['forecast_remaining_seconds']
        meter = type('Meter', (), {'rounds': [{'phase': 'retrain', 'wall_seconds': w} for w in first]})()
        for b, expected in ((budget*1.0001, True), (budget*.9999, False)):
            self.assertEqual(base._checkpoint_decision(meter, b, .25)['continued'], expected)

    def test_budget_arguments_are_checked_before_any_change(self):
        fed = self.unlearner(); fed.fit()
        request = ForgetRequest(0, self.parts[0][:3])
        for kwargs in ({}, {'budget_seconds': -1.}, {'budget_seconds': float('inf')}, {'budget_seconds': 5., 'margin': -1.}):
            with self.assertRaises(ValueError):
                fed.unlearn(request, 'shard_checkpoint', **kwargs)
        with self.assertRaises(ValueError):
            fed.unlearn(request, 'shard_full', budget_seconds=5.)
        self.assertTrue(fed.alive.all()); self.assertEqual(fed.step, 0)

    def test_checkpoint_policy_survives_save_and_load(self):
        d = self.data
        with tempfile.TemporaryDirectory() as tmp:
            fed = self.unlearner('shard_checkpoint'); fed.fit()
            result = fed.unlearn(ForgetRequest(2, self.parts[2][:4]), budget_seconds=0., checkpoint_dir=Path(tmp)/'s1')
            restored = ShardedUnlearner.load(Path(tmp)/'s1', d['train'], d['validation'], self.parts)
            self.assertEqual(restored.policy, 'shard_checkpoint')
            self.assertEqual(restored.shards[result.retrained_shard].rounds, FIXED)
            self.assertTrue(restored.verify_bit_exact()['bit_exact'])


class BatchTests(Base):
    def test_single_request_batch_trains_what_unlearn_trains(self):
        base = self.unlearner(); base.fit()
        request = ForgetRequest(3, self.parts[3][:7])
        for policy in ('shard_full', 'shard_fixed'):
            one, batch = base.fork(policy), base.fork(policy)
            a, b = one.unlearn(request), batch.unlearn_batch([request])
            j = a.retrained_shard
            self.assertEqual(b.retrained_shard, [j])
            self.assertTrue(weights_equal(one.shards[j].model, batch.shards[j].model))
            self.assertEqual(a.cost['cumulative_unlearning'], b.cost['cumulative_unlearning'])
            self.assertEqual((b.kind, b.client, b.rows, b.event), ('batch', 3, 7, 'retrain'))
            self.assertEqual(one.history[-1]['forget_request_hash'], batch.history[-1]['forget_request_hash'])

    def test_batch_retrains_each_touched_shard_once_and_ends_as_the_sequential_stream(self):
        base = self.unlearner(); base.fit()
        s0, s1 = self.assignment
        requests = [ForgetRequest(s0[0], self.parts[s0[0]][:6]), ForgetRequest(s1[0], self.parts[s1[0]][:4]),
                    ForgetRequest(s0[1], self.parts[s0[1]][:5]), ForgetRequest(s0[0], self.parts[s0[0]][6:9])]
        seq, batch = base.fork('shard_full'), base.fork('shard_full')
        for r in requests:
            seq.unlearn(r)
        result = batch.unlearn_batch(requests)
        self.assertEqual(result.details['shards_retrained'], [0, 1])
        self.assertEqual(result.cost['rounds'], 2*RB)
        self.assertEqual(result.cost['rows_x_rounds'], RB*int(batch.alive.sum()))      # both shards, final data, once each
        self.assertEqual(result.client, -1)
        self.assertEqual([h['step'] for h in batch.history], [1, 2, 3, 4])
        self.assertEqual({h['batch_first_step'] for h in batch.history}, {1})
        self.assertTrue(np.array_equal(seq.alive, batch.alive))
        for j in (0, 1):
            self.assertTrue(weights_equal(seq.shards[j].model, batch.shards[j].model))   # the final states are equal bit for bit
        self.assertLess(result.cost['rows_x_rounds'], seq.totals['rows_x_rounds'])
        self.assertTrue(batch.verify_bit_exact()['bit_exact'])

    def test_batch_follows_structure_changes(self):
        assignment = [[0, 1, 2, 3], [4, 5, 6, 7]]
        base = self.unlearner(assignment=assignment, merge_below=3); base.fit()
        requests = [ForgetRequest(0, self.parts[0][:3]), ForgetRequest(1), ForgetRequest(2), ForgetRequest(5, self.parts[5][:2])]
        seq, batch = base.fork('shard_full'), base.fork('shard_full')
        events = [seq.unlearn(r).event for r in requests]
        self.assertIn('merge_0_into_1', events)
        result = batch.unlearn_batch(requests)
        self.assertEqual((result.retrained_shard, result.retired_shards), ([1], [0]))
        self.assertEqual(result.event, 'merge_0_into_1')
        self.assertIn('STRUCTURE_CHANGED', {w['code'] for w in result.structured_warnings})
        self.assertEqual(batch.assignment, seq.assignment)
        self.assertTrue(weights_equal(seq.shards[1].model, batch.shards[1].model))
        self.assertIsNone(batch.shards[0])
        self.assertTrue(batch.verify_bit_exact()['bit_exact'])

    def test_invalid_batches_fail_before_any_change(self):
        fed = self.unlearner(); fed.fit()
        held = self.parts[0]
        bad = [[], [ForgetRequest(0, held[:3]), ForgetRequest(0, held[2:4])], [ForgetRequest(0), ForgetRequest(0, held[:1])],
               [ForgetRequest(1, held[:2])]]
        for requests in bad:
            with self.assertRaises(ValueError):
                fed.unlearn_batch(requests)
        for policy in ('utility_stop', 'drop_backfill', 'shard_checkpoint'):
            with self.assertRaises(ValueError):
                fed.unlearn_batch([ForgetRequest(0, held[:2])], policy)
        self.assertTrue(fed.alive.all()); self.assertEqual((fed.step, fed.history), (0, []))
        backlog = self.unlearner('drop_backfill', backfill_rounds=1); backlog.fit()
        backlog.unlearn(ForgetRequest(1, self.parts[1][:3]))
        with self.assertRaisesRegex(RuntimeError, 'backlog'):
            backlog.unlearn_batch([ForgetRequest(2, self.parts[2][:3])], 'shard_full')


class StressTests(Base):
    def test_stress_changes_time_not_weights(self):
        base = self.unlearner(); base.fit()
        request = ForgetRequest(0, self.parts[0][:8])
        plain = base.fork('shard_full'); plain.unlearn(request)
        slept = []
        stressed = base.fork('shard_full')
        slow = self.assignment[0][1]
        controller = install(stressed, lambda step: Condition(slow={slow: 3.}, bandwidth_bps=8e6), sleep=slept.append)
        controller.step = 1
        result = stressed.unlearn(request)
        j = result.retrained_shard
        self.assertTrue(weights_equal(plain.shards[j].model, stressed.shards[j].model))
        clients = len(self.assignment[0])
        self.assertEqual((len(slept), controller.workers_built), (RB*clients, clients))
        transfer = 2*stressed.model_bytes*8/8e6
        rows = [r for r in result.client_costs if r['client_id'] == slow]
        others = [s for s, r in zip(slept, result.client_costs) if r['client_id'] != slow]
        self.assertTrue(all(abs(s - transfer) < 1e-12 for s in others))
        for s, r in zip([s for s, r in zip(slept, result.client_costs) if r['client_id'] == slow], rows):
            self.assertAlmostEqual(s - transfer, 2/3*r['compute_seconds'])                      # (f - 1) x measured; reported f x
        self.assertTrue(stressed.verify_bit_exact()['bit_exact'])

    def test_real_sleep_reaches_the_round_wall_time(self):
        base = self.unlearner(); base.fit()
        fed = base.fork('shard_full')
        install(fed, lambda step: Condition(bandwidth_bps=2*fed.model_bytes*8/.01)).step = 1     # 10 ms per client and round
        result = fed.unlearn(ForgetRequest(0, self.parts[0][:8]))
        clients = len(self.assignment[0])
        for r in result.round_costs:
            self.assertGreaterEqual(r['wall_seconds'], .01*clients)
            self.assertGreaterEqual(r['server_seconds'], .01*clients*.9)                         # transfer is not compute

    def test_environments(self):
        assignment = assign_shards(4, 20)
        chosen = slow_clients(assignment)
        self.assertEqual(len(chosen), 4)
        self.assertTrue(all(sum(c in a for c in chosen) == 1 for a in assignment))
        none, slow, bw, transient = (environment(e, assignment) for e in stress.ENVIRONMENTS)
        on, end = stress.STRESS['onset'], stress.STRESS['transient_end']
        self.assertFalse(any(none(k).active for k in range(1, 21)))
        self.assertEqual([slow(k).active for k in (on - 1, on, 20)], [False, True, True])
        self.assertEqual([bw(k).bandwidth_bps for k in (on - 1, on)], [float('inf'), stress.STRESS['bandwidth_bps']])
        self.assertEqual([transient(k).active for k in (on - 1, on, end, end + 1)], [False, True, True, False])
        self.assertEqual(set(slow(on).slow), set(chosen))
        with self.assertRaises(ValueError):
            StressedWorker(object(), factor=.5)


def fake_request(step, counts, rate, server=.01, client=None, rows=1, shard=0, rounds=20):
    """A measured request: every client computes rate[c] x samples per round, the server adds `server`."""
    per_round = sum(rate[c]*n for c, n in counts.items()) + server
    return {'step': step, 'shard': shard, 'client': client if client is not None else next(iter(counts)), 'rows': rows,
            'counts': {str(c): n for c, n in counts.items()},
            'rounds': [{'round': r, 'shard': shard, 'samples': sum(counts.values()), 'wall_seconds': per_round,
                        'server_seconds': server} for r in range(1, rounds + 1)],
            'clients': [{'round': r, 'client_id': c, 'samples': n, 'compute_seconds': rate[c]*n}
                        for r in range(1, rounds + 1) for c, n in counts.items()]}


class PlannerTests(Base):
    def setUp(self):
        super().setUp()
        from src.dt import planner
        self.p = planner
        self.counts = {0: 1000, 1: 500}
        self.profile = {'client': {'0': 1e-3, '1': 1e-3}, 'server': .01}         # 1.51 s per round as calibrated
        self.history = ([1.], [1.])

    def test_forecasts_equal_the_existing_twin(self):
        from scripts.dt.run_stress_trajectories import shard_counts
        from scripts.shard import shard_dt
        fed = self.unlearner(); fit = fed.fit()
        profile = shard_dt.rates(fit['round_costs'], fit['client_costs'])
        self.assertEqual(self.p.rates(fit['round_costs'], fit['client_costs']), profile)
        c = self.assignment[1][2]
        request = ForgetRequest(c, self.parts[c][:9])
        report = shard_dt.plan(fed, request, profile, 1e9, 10**12, True)
        counts = shard_counts(fed, 1, c, 9)
        for cand in report['candidates']:
            self.assertAlmostEqual(cand['seconds'], self.p.forecast(profile, counts, cand['rounds']))
        ratios, extra = self.p.fit_ratios(fit['round_costs'], fit['client_costs'], profile)
        self.assertEqual((len(ratios), len(extra)), (2, 2))
        self.assertTrue(all(.5 < r < 2 for r in ratios))

    def test_oracle_never_violates_or_rejects_wrongly(self):
        rng = np.random.default_rng(0)
        requests = [fake_request(k + 1, self.counts, {0: rng.uniform(5e-4, 4e-3), 1: 1e-3}) for k in range(30)]
        for budget in (10., 30., 60.):
            s = self.p.summarize(self.p.simulate(requests, budget, self.p.PlannerConfig('oracle'), self.profile, self.history))
            self.assertEqual((s['violations'], s['false_rejects'], s['downgrades']), (0, 0, 0))
            self.assertEqual(s['accepted'], s['oracle_accepted'])

    def test_margin_learns_from_past_errors(self):
        requests = [fake_request(k + 1, self.counts, {0: 2e-3, 1: 2e-3}) for k in range(12)]   # twice as slow as calibrated
        budget = 40.                                     # forecast full 30.2 s fits; true full 60.2 s does not; true fixed 30.1 s does
        plain = self.p.simulate(requests, budget, self.p.PlannerConfig('static'), self.profile, self.history)
        margin = self.p.simulate(requests, budget, self.p.PlannerConfig('static', .9), self.profile, self.history)
        self.assertTrue(all(r['violation'] and r['choice'] == 'full' for r in plain))
        self.assertTrue(margin[0]['violation'])          # nothing learned yet
        self.assertAlmostEqual(margin[1]['factor'], 1 + .9*(60.2/30.2 - 1))  # q90 of the history [1, 60.2/30.2]
        self.assertTrue(all(r['choice'] == 'fixed' and not r['violation'] for r in margin[1:]))
        adaptive = self.p.simulate(requests, budget, self.p.PlannerConfig('adaptive'), self.profile, self.history)
        self.assertGreater(adaptive[1]['forecast_full'], adaptive[0]['forecast_full'])

    def test_rescored_margin_forgets_an_error_the_profile_has_absorbed(self):
        requests = ([fake_request(k + 1, self.counts, {0: 1e-3, 1: 1e-3}) for k in range(3)]
                    + [fake_request(k + 4, self.counts, {0: 3e-3, 1: 3e-3}) for k in range(12)])   # three times slower from step 4
        ratio = self.p.simulate(requests, 1e9, self.p.PlannerConfig('adaptive', .9), self.profile, self.history)
        rescored = self.p.simulate(requests, 1e9, self.p.PlannerConfig('adaptive', .9, margin_mode='rescored'), self.profile, self.history)
        self.assertEqual([r['factor'] for r in ratio[:4]], [r['factor'] for r in rescored[:4]])
        self.assertGreater(ratio[-1]['factor'], 1.5)                         # the spikes after the change are still in the window
        self.assertLess(rescored[-1]['factor'], 1.1)                         # the adapted profile explains the recent operations
        self.assertLess(abs(rescored[-1]['forecast_full']/requests[-1]['rounds'][0]['wall_seconds']/20 - 1), .05)
        static = self.p.simulate(requests, 1e9, self.p.PlannerConfig('static', .9, margin_mode='rescored'), self.profile, self.history)
        same = self.p.simulate(requests, 1e9, self.p.PlannerConfig('static', .9), self.profile, self.history)
        self.assertEqual([r['factor'] for r in static], [r['factor'] for r in same])

    def fit_records(self, shards, rate, server=.01):
        """Cost records of a base fit: shards = {shard: {client: samples}}, every client at `rate` seconds per sample."""
        rounds, clients = [], []
        for j, counts in shards.items():
            req = fake_request(0, counts, {c: rate for c in counts}, server, shard=j)
            rounds += req['rounds']; clients += [{**c, 'shard': j} for c in req['clients']]
        return rounds, clients

    def test_simple_comparators(self):
        p = self.p
        shards = {0: {0: 1000, 1: 500}, 1: {2: 4000, 3: 4000}}
        fit = self.fit_records(shards, 1e-3)
        profile = p.rates(*fit); history = p.fit_ratios(*fit, profile)
        run = lambda name, requests, budget=1e9, **kw: p.simulate(requests, budget, p.PlannerConfig(name, **kw), profile, history, fit)
        slow = {c: 3e-3 for c in range(4)}
        requests = [fake_request(1, shards[0], slow, shard=0), fake_request(2, shards[1], slow, shard=1),
                    fake_request(3, shards[0], slow, shard=0), fake_request(4, {2: 4000, 3: 2000}, slow, client=3, shard=1)]
        truth = [p.cost(r, 20) for r in requests]
        # last_shard: wrong on the first visit of each shard after the change, exact on the next one, and scaled by the rows
        last = run('last_shard', requests)
        self.assertAlmostEqual(last[0]['forecast_full'], 20*(1.5 + .01)); self.assertAlmostEqual(last[1]['forecast_full'], 20*(8 + .01))
        self.assertAlmostEqual(last[2]['forecast_full'], truth[2])
        self.assertAlmostEqual(last[3]['forecast_full'], truth[1]*6000/8000)
        self.assertEqual(p.Twin(p.PlannerConfig('last_shard', .9), profile, history, fit).initial, [1., 1.])
        # rows_static: one rate for everything, never updated; rows_ewma moves 30 % of the way per observation
        rate = sum(r['wall_seconds'] for r in fit[0])/sum(r['samples'] for r in fit[0])
        static = run('rows_static', requests)
        self.assertEqual([round(r['forecast_full'], 9) for r in static], [round(20*rate*n, 9) for n in (1500, 8000, 1500, 6000)])
        moving = run('rows_ewma', requests)
        seen = truth[0]/20/1500
        self.assertAlmostEqual(moving[1]['forecast_full'], 20*(.7*rate + .3*seen)*8000)
        # the fixed policies decide nothing: no rejection, a violation whenever the option does not fit
        budget = (p.cost(requests[1], 10) + truth[1])/2            # shard 1: shard_fixed fits, shard_full does not
        full, fixed = run('always_full', requests, budget), run('always_fixed', requests, budget)
        self.assertEqual([r['choice'] for r in full], ['full']*4); self.assertEqual([r['choice'] for r in fixed], ['fixed']*4)
        self.assertEqual([r['violation'] for r in full], [t > budget for t in truth])
        self.assertEqual([r['violation'] for r in full][:3], [False, True, False])
        self.assertEqual([r['violation'] for r in fixed], [False]*4)
        self.assertEqual([r['downgrade'] for r in fixed], [t <= budget for t in truth])
        self.assertEqual((p.summarize(full)['rejected'], p.summarize(fixed)['false_rejects']), (0, 0))
        self.assertTrue(np.isnan(p.summarize(full)['forecast_mae_seconds']))
        with self.assertRaisesRegex(ValueError, 'fit_records'):
            p.simulate(requests, 1e9, p.PlannerConfig('last_shard'), profile, history)
        with self.assertRaises(ValueError):
            p.make_predictor('nope', profile, fit[0])
        # the twin's own predictors do not need the fit records and are unchanged by them
        self.assertEqual(run('adaptive', requests, margin_q=.9), p.simulate(requests, 1e9, p.PlannerConfig('adaptive', .9), profile, history))

    def test_telemetry_summary_is_sufficient_for_the_twin(self):
        from src.dt import overhead
        rng = np.random.default_rng(1)
        req = fake_request(1, {0: 1000, 1: 500, 7: 40}, {0: 1e-3, 1: 2e-3, 7: 5e-3})
        for c in req['clients']:                                  # every client-round with its own time
            c['compute_seconds'] *= rng.uniform(.5, 2)
        rounds, clients = self.p.observed_records(req, 20)
        summary = overhead.operation_summary(rounds, clients)
        want, got = self.p.rates(rounds, clients), overhead.summary_rates(summary)
        self.assertEqual(set(want['client']), set(got['client']))
        for c in want['client']:
            self.assertAlmostEqual(want['client'][c], got['client'][c])
        self.assertAlmostEqual(want['server'], got['server'])
        self.assertAlmostEqual(summary['wall_seconds_until_checkpoint'], self.p.cost(req, 10))
        self.assertAlmostEqual(summary['wall_seconds'], self.p.cost(req, 20))
        sizes = overhead.telemetry_bytes(rounds, clients)
        self.assertLess(sizes['summary_json'], sizes['records_zlib'])
        self.assertLess(sizes['records_zlib'], sizes['records_json'])
        timed = overhead.time_twin([req, fake_request(2, {0: 990, 1: 500, 7: 40}, {0: 1e-3, 1: 2e-3, 7: 5e-3})],
                                   self.p.PlannerConfig('adaptive', .9, margin_mode='rescored'),
                                   {'client': {'0': 1e-3, '1': 1e-3, '7': 1e-3}, 'server': .01}, self.history, repeats=3)
        self.assertEqual([t['step'] for t in timed], [1, 2])
        self.assertTrue(all(0 < t['plan_seconds'] < .1 and 0 < t['sync_seconds'] < .1 for t in timed))

    def test_midcourse_stops_on_underforecast_and_continues_on_overforecast(self):
        slow = [fake_request(1, self.counts, {0: 2e-3, 1: 2e-3})]
        fast = [fake_request(1, self.counts, {0: 5e-4, 1: 5e-4})]
        pre, mid = self.p.PlannerConfig('static'), self.p.PlannerConfig('static', mode='midcourse')
        under_pre = self.p.simulate(slow, 40., pre, self.profile, self.history)[0]
        under_mid = self.p.simulate(slow, 40., mid, self.profile, self.history)[0]
        self.assertEqual((under_pre['choice'], under_pre['violation']), ('full', True))
        self.assertEqual((under_mid['choice'], under_mid['violation'], under_mid['executed_rounds']), ('fixed', False, 10))
        over_pre = self.p.simulate(fast, 20., pre, self.profile, self.history)[0]      # forecast full 30.2 > 20; true full 15.2
        over_mid = self.p.simulate(fast, 20., mid, self.profile, self.history)[0]
        self.assertEqual((over_pre['choice'], over_pre['downgrade']), ('fixed', True))
        self.assertEqual((over_mid['choice'], over_mid['violation'], over_mid['downgrade']), ('full', False, False))
        self.assertEqual(self.p.simulate(fast, 5., mid, self.profile, self.history)[0]['choice'], 'reject')

    def test_lag_delays_observations_and_ages_the_counts(self):
        requests = [fake_request(k + 1, {0: 1000 - 10*k, 1: 500}, {0: 2e-3, 1: 2e-3}, client=0, rows=10) for k in range(6)]
        fresh = self.p.simulate(requests, 1e9, self.p.PlannerConfig('adaptive'), self.profile, self.history)
        stale = self.p.simulate(requests, 1e9, self.p.PlannerConfig('adaptive', lag=3), self.profile, self.history)
        static = self.p.simulate(requests, 1e9, self.p.PlannerConfig('static'), self.profile, self.history)
        self.assertGreater(fresh[1]['forecast_full'], static[1]['forecast_full'])
        self.assertEqual(self.p.twin_counts(requests, 4, 3), {0: 1000 - 10*4 + 30, 1: 500})
        for k in (1, 2, 3):                               # the first observation arrives before request index 4
            self.assertAlmostEqual(stale[k]['forecast_full'], self.p.forecast(self.profile, self.p.twin_counts(requests, k, 3), 20))
        self.assertGreater(stale[4]['forecast_full'], stale[3]['forecast_full'])


class SchedulerTests(unittest.TestCase):
    ROWS = [4768, 1308, 6095, 6102, 3364, 4508, 2490, 12280, 1056, 4445,
            29317, 4278, 15035, 7034, 814, 1794, 85, 5144, 317, 4050]     # partition 42 (test_stream_v3.py)

    def counting_executor(self, assignment, seconds_per_row_round=1e-5):
        n = np.array(self.ROWS)

        def execute(batch, policy):
            for r in batch:
                n[r['client']] -= r['rows']
            rounds = 20 if policy == 'shard_full' else 10
            cost = sum(int(n[assignment[j]].sum()) for j in sorted({r['shard'] for r in batch}))*rounds
            return {'seconds': cost*seconds_per_row_round, 'rows_x_rounds': cost}
        return execute, n

    def test_window_and_immediate_modes_reproduce_the_v3_cost_plan(self):
        from scripts.shard.stream_v3 import FAMILIES, cost_plan, plan_stream
        from src.dt import scheduler
        A = assign_shards(4, 20)
        for fam in FAMILIES:
            stream = plan_stream(fam, self.ROWS, A)
            for W in (0, 2, 5):
                execute, _ = self.counting_executor(A)
                events = scheduler.run(stream, 'immediate' if W == 0 else 'window', 5., 1., execute, window=W or None)
                s, plan = scheduler.summarize(events, 5.), cost_plan(stream, A, self.ROWS, 20, W)
                self.assertEqual((s['rows_x_rounds'], s['retrains'], s['events']),
                                 (plan['shard_rows_x_rounds'], plan['shard_retrains'], plan['batches']), (fam, W))
                self.assertAlmostEqual(s['latency_mean'], plan['latency_mean'])
                self.assertAlmostEqual(s['latency_max'], plan['latency_max'])
                self.assertEqual(s['requests_served'], 20)

    def test_lazy_mode_waits_for_the_deadline_per_shard(self):
        from scripts.shard.stream_v3 import plan_stream
        from src.dt import scheduler
        A = assign_shards(4, 20)
        stream = plan_stream('concentrated', self.ROWS, A)
        execute, _ = self.counting_executor(A)
        events = scheduler.run(stream, 'lazy', 5., 1., execute)
        s = scheduler.summarize(events, 5.)
        self.assertEqual(s['deadline_misses'], 0)
        self.assertAlmostEqual(max(max(e['latencies']) for e in events), 5.)
        self.assertTrue(all(len(e['shards']) == 1 for e in events))
        self.assertLess(s['retrains'], 20)

    def test_dt_mode_serves_when_affordable_and_forces_at_the_deadline(self):
        from src.dt import scheduler

        class Twin:                         # forecast: 1 s per 20 rounds on shard 0, 3 s on shard 1; no margin
            def __init__(self): self.seen = []
            def forecast(self, shard, queued, rounds): return (1. if shard == 0 else 3.)*rounds/20
            def factor(self): return 1.
            def observe(self, result, predicted): self.seen.append((result['seconds'], predicted))

        stream = [{'step': 1, 'time': .5, 'shard': 0, 'rows': 1}, {'step': 2, 'time': 1., 'shard': 1, 'rows': 1},
                  {'step': 3, 'time': 2., 'shard': 1, 'rows': 1}, {'step': 4, 'time': 3., 'shard': 1, 'rows': 1},
                  {'step': 5, 'time': 10., 'shard': 1, 'rows': 1}]
        executed = []

        def execute(batch, policy):
            executed.append(([r['step'] for r in batch], policy))
            return {'seconds': (1. if batch[0]['shard'] == 0 else 3.)*(1 if policy == 'shard_full' else .5)}
        twin = Twin()
        events = scheduler.run(stream, 'dt', 5., 1.2, execute, twin=twin)
        # shard 0 fits alone; shard 1 needs 3 requests (3 x 1.2 >= 3); the lone last request is forced to shard_fixed
        self.assertEqual(executed, [([1], 'shard_full'), ([2, 3, 4], 'shard_full'), ([5], 'shard_fixed')])
        self.assertEqual([e['time'] for e in events], [.5, 3., 15.])
        self.assertEqual([e['forced_over_budget'] for e in events], [False, False, True])
        self.assertEqual([e['over_budget'] for e in events], [False, False, True])
        self.assertEqual(len(twin.seen), 3)
        self.assertEqual(scheduler.summarize(events, 5.)['deadline_misses'], 0)
        with self.assertRaises(ValueError):
            scheduler.run(stream, 'dt', 5., 1., execute)


TINY = {'fl': {'rounds': 4, 'local_epochs': 1, 'batch_size': 128, 'lr': 1e-2}, 'unlearn_budgets': [1, 4],
        'shard_fixed_rounds': 2, 'backfill_rounds': 3}


def small_stream():
    from scripts.shard import stream_v3
    return mock.patch.dict(stream_v3.STREAM, K=8, size_median=20)


class TrajectoryRunnerTests(TorchCase):
    def test_runner_writes_paired_costs_of_both_options(self):
        import contextlib, io
        from scripts.dt import run_stress_trajectories as runner
        from scripts.shard import train_config
        data = synthetic(4000, seed=4)
        with tempfile.TemporaryDirectory() as tmp, small_stream(), mock.patch.dict(train_config.CONFIGS, tiny=TINY), \
                mock.patch.dict(stress.STRESS, bandwidth_bps=1e12, onset=3), \
                mock.patch('src.data.loading.load_data', lambda path: data), contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)/'run'
            runner.main([str(out), '--seed', '7', '--family', 'spread', '--envs', 'none', 'slow_clients', '--train-config', 'tiny'])
            run = json.loads((out/'run.json').read_text())
            envs = {e: [json.loads(line) for line in (out/e/'requests.jsonl').read_text().splitlines()] for e in run['envs']}
        self.assertEqual(len(run['requests']), 8)
        self.assertEqual([r['shard'] for r in run['requests']][:4], [0, 1, 2, 3])
        self.assertGreater(run['fit']['largest_shard_full_seconds'], run['fit']['largest_shard_fixed_seconds'])
        for env, rows in envs.items():
            self.assertEqual([r['step'] for r in rows], list(range(1, 9)))
            for r in rows:
                self.assertEqual([x['round'] for x in r['rounds']], list(range(1, 5)))
                self.assertEqual(set(r['counts']), {str(c) for c in run['assignment'][r['shard']]})
                self.assertTrue(r['exact'] and r['structure_ok'])
            self.assertEqual(envs['none'][2]['counts'], rows[2]['counts'])
        slow = [r['condition']['slow'] for r in envs['slow_clients']]
        self.assertEqual([bool(s) for s in slow], [False, False] + [True]*6)
        self.assertEqual([r['validation_macro_f1'] for r in envs['none']], [r['validation_macro_f1'] for r in envs['slow_clients']])

    def test_overhead_script_measures_the_twin_on_trajectories_and_live(self):
        import contextlib, io
        import pandas as pd
        from scripts.dt import measure_dt_overhead as script, run_stress_trajectories as runner
        from scripts.shard import train_config
        data = synthetic(4000, seed=4)
        with tempfile.TemporaryDirectory() as tmp, small_stream(), mock.patch.dict(train_config.CONFIGS, tiny=TINY), \
                mock.patch('src.data.loading.load_data', lambda path: data), contextlib.redirect_stdout(io.StringIO()):
            runner.main([str(Path(tmp)/'runs'/'a'), '--seed', '7', '--family', 'spread', '--envs', 'none', '--train-config', 'tiny'])
            out = Path(tmp)/'overhead'
            script.main([str(Path(tmp)/'runs'), str(out), '--repeats', '2', '--live', '--seed', '7', '--requests', '4',
                         '--train-config', 'tiny'])
            times, telemetry = pd.read_csv(out/'twin_times.csv'), pd.read_csv(out/'telemetry.csv')
            summary, live = json.loads((out/'summary.json').read_text()), json.loads((out/'live.json').read_text())
        self.assertEqual(sorted(times.twin.unique()), sorted(script.TWINS))
        self.assertEqual(len(times), 8*len(script.TWINS))
        self.assertTrue((times.plan_seconds > 0).all() and (times.sync_seconds > 0).all())
        self.assertEqual(sorted(telemetry.option.unique()), ['fixed', 'full'])
        self.assertTrue((telemetry.summary_json < telemetry.records_json).all())
        self.assertTrue((telemetry.records_json < telemetry.payload_bytes_raw).all())
        self.assertLess(summary['twins']['adaptive']['dt_share_of_fu_total'], 1)
        self.assertEqual(set(live['state']['shard_counts']), {'0', '1', '2', '3'})
        self.assertEqual(live['state']['client_state']['clients'], 20)
        self.assertTrue(live['metering']['weights_identical'])                # measuring the payload never changes a weight
        self.assertEqual(live['metering']['requests'], 4)
        self.assertGreater(live['metering']['payload_pack_seconds_total'], 0)

    def test_online_midcourse_runner_drives_the_checkpoint_policy(self):
        import contextlib, io
        from scripts.dt import run_midcourse_online as runner
        from scripts.shard import train_config
        from src.dt import planner
        data = synthetic(4000, seed=4)
        with tempfile.TemporaryDirectory() as tmp, small_stream(), mock.patch.dict(train_config.CONFIGS, tiny=TINY), \
                mock.patch.dict(planner.BUDGET_LEVELS, wide=1e4), mock.patch.dict(stress.STRESS, onset=3), \
                mock.patch('src.data.loading.load_data', lambda path: data), contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)/'run'
            runner.main([str(out), '--seed', '7', '--family', 'spread', '--env', 'slow_clients', '--budget-level', 'wide',
                         '--train-config', 'tiny', '--margin-mode', 'rescored'])
            rows = [json.loads(line) for line in (out/'decisions.jsonl').read_text().splitlines()]
            final = json.loads((out/'final.json').read_text())
            run = json.loads((out/'run.json').read_text())
        self.assertEqual(len(rows), 8)
        self.assertTrue(all(r['policy'] == 'shard_checkpoint' and r['choice'] == 'full' for r in rows))
        self.assertTrue(all(r['exactness_scope'] == 'unconditional' and r['exact'] for r in rows))
        self.assertTrue(final['verify_bit_exact']['bit_exact'])
        self.assertEqual(run['planner'], 'adaptive_margin_rescored_midcourse')
        # the twin's own cost is recorded around every request and summed at the end
        self.assertTrue(all(set(r['dt_seconds']) == {'state', 'plan', 'sync'} and r['dt_overhead_seconds'] > 0 for r in rows))
        self.assertTrue(all(r['telemetry_bytes']['summary_json'] < r['telemetry_bytes']['records_json'] < r['payload_bytes_raw']
                            for r in rows))
        self.assertAlmostEqual(final['dt_overhead']['dt_seconds'], sum(r['dt_overhead_seconds'] for r in rows))
        self.assertAlmostEqual(final['dt_overhead']['fu_round_seconds'], sum(r['seconds'] for r in rows))

    def test_batch_runner_ends_every_shard_full_schedule_in_the_same_state(self):
        import contextlib, io
        import pandas as pd
        from scripts.dt import run_batch_dt as runner
        from scripts.shard import train_config
        data = synthetic(4000, seed=4)
        with tempfile.TemporaryDirectory() as tmp, small_stream(), mock.patch.dict(train_config.CONFIGS, tiny=TINY), \
                mock.patch('src.data.loading.load_data', lambda path: data), contextlib.redirect_stdout(io.StringIO()):
            out = Path(tmp)/'run'
            runner.main([str(out), '--seed', '7', '--family', 'concentrated', '--train-config', 'tiny', '--verify'])
            summary = pd.read_csv(out/'summary.csv').set_index('mode')
            finals = {m: json.loads((out/m/'final.json').read_text()) for m in runner.MODES}
            events = {m: [json.loads(line) for line in (out/m/'events.jsonl').read_text().splitlines()] for m in runner.MODES}
            from scripts.dt import analyze_batch
            analyze_batch.main([tmp, str(Path(tmp)/'analysis')])
            states = pd.read_csv(Path(tmp)/'analysis'/'final_states.csv')
            table = pd.read_csv(Path(tmp)/'analysis'/'summary.csv').set_index('mode')
        self.assertTrue(states.final_states_equal.all() and states.bit_exact.all() and states.bit_exact_checked.all())
        self.assertEqual(table.loc['immediate', 'cost_vs_immediate'], 1.)
        self.assertEqual(finals['immediate']['weights_sha256'], finals['window5']['weights_sha256'])
        self.assertEqual(finals['immediate']['weights_sha256'], finals['lazy']['weights_sha256'])
        self.assertTrue(all(f['verify_bit_exact']['bit_exact'] and f['exact'] and f['structure_ok'] for f in finals.values()))
        self.assertTrue(summary.exact_all_events.all())
        self.assertEqual(set(summary.requests_served), {8})
        self.assertEqual(summary.loc['immediate', 'events'], 8)
        self.assertLess(summary.loc['window5', 'rows_x_rounds'], summary.loc['immediate', 'rows_x_rounds'])
        self.assertTrue((summary.deadline_misses == 0).all())
        self.assertTrue(all(e['forecast_seconds'] is not None for e in events['dt_mid']))


if __name__ == '__main__':
    unittest.main()
