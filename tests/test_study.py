import copy
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch import nn

from src.data.scenarios import exclusive_family_partition, family_request, unique_feature_rows, atypical_request
from src.eval.lira import logit_confidence, out_statistics, lira_scores, attack_summary, near_duplicate_index, near_duplicates
from src.eval.metrics import matched_indices
from src.eval.noise_floor import retrain_noise, distance_to_retrains, margin, identifiable
from src.fl.fedavg_core import ClientWorker, batches, model
from src.fu.baselines import retain_finetune
from src.fu.conpfu_inspired import alternating, distill, forget_loss, forget_step, gradient_batches, mean_ce
from src.fu.recovery_check import forget_with_recovery_check

LEGACY = {'cycles': 1, 'batch_size': 16, 'forget_lr': .0005, 'retain_lr': .0005, 'regularization': .001,
          'clip': 1., 'kd_epochs': 1, 'kd_lr': .001, 'temperature': 2., 'hard_weight': .5}


def pilot_alternating(original, forget, retain, config, seed):
    """Verbatim pilot implementation, kept as the reference for backward compatibility."""
    def step(model, x, y, anchor, lr, regularization, clip):
        optimizer = torch.optim.SGD(model.parameters(), lr=lr)
        optimizer.zero_grad()
        ce = nn.functional.cross_entropy(model(x), y)
        penalty = sum((p - a).square().sum() for p, a in zip(model.parameters(), anchor))
        (-ce + regularization * penalty).backward()
        nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
        optimizer.step()
    local = copy.deepcopy(original)
    anchor = [p.detach().clone() for p in local.parameters()]
    for cycle in range(config['cycles']):
        for idx in batches(len(forget.y), config['batch_size'], seed + cycle):
            step(local, forget.x[idx], forget.y[idx], anchor, config['forget_lr'], config['regularization'], config['clip'])
        optimizer = torch.optim.SGD(local.parameters(), lr=config['retain_lr'])
        for idx in batches(len(retain.y), config['batch_size'], seed + cycle):
            optimizer.zero_grad()
            nn.functional.cross_entropy(local(retain.x[idx]), retain.y[idx]).backward()
            nn.utils.clip_grad_norm_(local.parameters(), config['clip'], error_if_nonfinite=True)
            optimizer.step()
    return local


def pilot_distill(original, teacher, retain, config, seed):
    teacher = copy.deepcopy(teacher).eval()
    teacher.requires_grad_(False)
    student = copy.deepcopy(original).train()
    optimizer = torch.optim.SGD(student.parameters(), lr=config['kd_lr'])
    for epoch in range(config['kd_epochs']):
        for idx in batches(len(retain.y), config['batch_size'], seed + epoch):
            optimizer.zero_grad()
            s, t = student(retain.x[idx]), teacher(retain.x[idx]).detach()
            soft = nn.functional.kl_div(nn.functional.log_softmax(s / config['temperature'], dim=1),
                                        nn.functional.softmax(t / config['temperature'], dim=1),
                                        reduction='batchmean') * config['temperature']**2
            (config['hard_weight'] * nn.functional.cross_entropy(s, retain.y[idx])
             + (1 - config['hard_weight']) * soft).backward()
            optimizer.step()
    return student


def toy(seed=0, n=120):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, 4)).astype(np.float32)
    y = (x[:, 0] + .3 * rng.normal(size=n) > 0).astype(np.int64)
    return x, y


def trained(x, y, seed=0, steps=200):
    m = model(x.shape[1], [8, 4], seed)
    optimizer = torch.optim.Adam(m.parameters(), lr=.01)
    xt, yt = torch.as_tensor(x), torch.as_tensor(y)
    for _ in range(steps):
        optimizer.zero_grad()
        nn.functional.cross_entropy(m(xt), yt).backward()
        optimizer.step()
    return m


def params_equal(a, b):
    return all(torch.equal(p, q) for p, q in zip(a.parameters(), b.parameters()))


def params_close(a, b, atol=1e-6):
    return all(torch.allclose(p, q, atol=atol) for p, q in zip(a.parameters(), b.parameters()))


class ForgetStageTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        x, y = toy()
        self.original = trained(x, y)
        self.forget = ClientWorker(0, x[:40], y[:40], np.arange(40))
        self.retain = ClientWorker(0, x[40:], y[40:], np.arange(40, 120))

    def test_legacy_config_reproduces_pilot_exactly(self):
        new, logs = alternating(self.original, self.forget, self.retain, LEGACY, 7)
        self.assertTrue(params_equal(new, pilot_alternating(self.original, self.forget, self.retain, LEGACY, 7)))
        self.assertEqual(gradient_batches(logs), 3 + 5)
        self.assertTrue(params_equal(self.original, trained(*toy())))  # original untouched

    def test_legacy_retain_kd_matches_pilot_kd(self):
        teacher, _ = alternating(self.original, self.forget, self.retain, LEGACY, 7)
        a, losses = distill(self.original, teacher, self.retain, LEGACY, 3)
        self.assertTrue(params_equal(a, pilot_distill(self.original, teacher, self.retain, LEGACY, 3)))
        self.assertEqual(len(losses), 5)

    def test_bounded_objectives(self):
        logits = torch.tensor([[3., -1.], [0., 0.], [-2., 4.]])
        y = torch.tensor([0, 1, 1])
        self.assertAlmostEqual(forget_loss(torch.zeros(3, 2), y, 'uniform').item(), math.log(2), places=6)
        self.assertGreaterEqual(forget_loss(logits, y, 'uniform').item(), math.log(2))
        flipped = nn.functional.cross_entropy(logits, 1 - y)
        self.assertAlmostEqual(forget_loss(logits, y, 'random_label').item(), flipped.item(), places=6)
        self.assertAlmostEqual(forget_loss(logits, y, 'neg_ce').item(),
                               -nn.functional.cross_entropy(logits, y).item(), places=6)
        with self.assertRaises(ValueError):
            forget_loss(logits, y, 'unknown')

    def test_neggrad_plus_needs_retain_and_counts_two_batches(self):
        m = copy.deepcopy(self.original)
        anchor = [p.detach().clone() for p in m.parameters()]
        with self.assertRaises(ValueError):
            forget_step(m, self.forget.x, self.forget.y, anchor, .01, 0, 1., objective='neggrad_plus')
        log = forget_step(m, self.forget.x, self.forget.y, anchor, .01, 0, 1., objective='neggrad_plus',
                          retain=(self.retain.x, self.retain.y))
        self.assertEqual(log['batches'], 2)

    def test_stop_rule_requires_target_and_stops_at_it(self):
        config = {**LEGACY, 'cycles': 20, 'forget_objective': 'uniform', 'forget_optimizer': 'adam', 'forget_lr': .01,
                  'retain_optimizer': 'adam', 'retain_lr': .001, 'retain_steps_per_cycle': 1, 'stop_rule': 'nonmember'}
        with self.assertRaises(ValueError):
            alternating(self.original, self.forget, self.retain, config, 1)
        start = mean_ce(self.original, self.forget.x, self.forget.y)
        local, logs = alternating(self.original, self.forget, self.retain, config, 1, start + .2)
        self.assertTrue(logs[-1]['stop_reached'])
        self.assertGreaterEqual(mean_ce(local, self.forget.x, self.forget.y), start + .2)
        self.assertLess(sum(e['phase'] == 'forget' for e in logs), 20 * 3)

    def test_stop_target_already_met_skips_forget_phase(self):
        config = {**LEGACY, 'forget_objective': 'neg_ce', 'stop_rule': 'nonmember'}
        _, logs = alternating(self.original, self.forget, self.retain, config, 1, 0.)
        self.assertEqual(sum(e['phase'] == 'forget' for e in logs), 0)
        self.assertEqual(gradient_batches(logs), 5)

    def test_adam_forget_moves_far_more_than_pilot_sgd(self):
        def moved(config):
            local, _ = alternating(self.original, self.forget, self.retain, {**config, 'retain_lr': 0.}, 1)
            return sum((p - q).square().sum() for p, q in zip(local.parameters(), self.original.parameters())).sqrt().item()
        adam = {**LEGACY, 'forget_objective': 'uniform', 'forget_optimizer': 'adam', 'forget_lr': .001}
        self.assertGreater(moved(adam), 3 * moved(LEGACY))


class RecoveryCheckTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        x, y = toy(3)
        self.original = trained(x, y, 3)
        self.forget = ClientWorker(0, x[:40], y[:40], np.arange(40))
        self.retain = ClientWorker(0, x[40:], y[40:], np.arange(40, 120))
        self.config = {**LEGACY, 'forget_objective': 'uniform', 'forget_optimizer': 'adam', 'forget_lr': .01,
                       'retain_steps_per_cycle': 1, 'stop_rule': 'proxy_calibrated', 'max_outer': 6}

    def test_check_runs_on_recovered_model(self):
        calls = []
        def undo(m, step):  # a "recovery" that always restores the original: never reaches the target
            calls.append(step)
            return copy.deepcopy(self.original)
        target = mean_ce(self.original, self.forget.x, self.forget.y) + .1
        final, first, logs = forget_with_recovery_check(self.original, self.forget, self.retain, self.config, 0, target, undo)
        self.assertEqual(calls, list(range(6)))
        self.assertFalse(logs[-1]['stop_reached'])
        self.assertTrue(params_equal(final, self.original))
        self.assertFalse(params_equal(first, self.original))  # the first local cycle did move the model
        self.assertEqual(logs[-1]['gradient_batches'], sum(e['local_gradient_batches'] for e in logs[:-1]))

    def test_stops_once_recovered_model_reaches_target(self):
        target = mean_ce(self.original, self.forget.x, self.forget.y) + .1
        final, _, logs = forget_with_recovery_check(self.original, self.forget, self.retain, self.config, 0, target,
                                                    lambda m, step: m)
        self.assertTrue(logs[-1]['stop_reached'])
        self.assertLess(logs[-1]['outer_steps'], 6)
        self.assertGreaterEqual(mean_ce(final, self.forget.x, self.forget.y), target)

    def test_rejects_kd_and_missing_target(self):
        with self.assertRaises(ValueError):
            forget_with_recovery_check(self.original, self.forget, self.retain, self.config, 0, None, lambda m, s: m)
        with self.assertRaises(ValueError):
            forget_with_recovery_check(self.original, self.forget, self.retain, {**self.config, 'kd_mode': 'full_local'},
                                       0, 1., lambda m, s: m)

    def test_anchor_model_pulls_back_to_original(self):
        start = copy.deepcopy(self.original)
        with torch.no_grad():
            for p in start.parameters():
                p.add_(.5)
        config = {**LEGACY, 'regularization': 10., 'forget_lr': .001, 'retain_lr': 0., 'clip': 1e6}
        dist = lambda m: sum((p - q).square().sum() for p, q in zip(m.parameters(), self.original.parameters())).item()
        anchored, _ = alternating(start, self.forget, self.retain, config, 0, anchor_model=self.original)
        free, _ = alternating(start, self.forget, self.retain, config, 0)
        self.assertLess(dist(anchored), dist(start))
        self.assertGreater(dist(anchored), 0)
        self.assertLess(dist(anchored), dist(free))


class DistillationTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)
        x, y = toy(1)
        x[:40, 2] += 6  # F is a separate region, as in family or atypical deletion
        self.original = trained(x, y, 1)
        self.forget = ClientWorker(0, x[:40], y[:40], np.arange(40))
        self.retain = ClientWorker(0, x[40:], y[40:], np.arange(40, 120))
        self.config = {**LEGACY, 'kd_epochs': 20, 'kd_lr': .01, 'kd_optimizer': 'adam'}

    def confidence(self, m, worker):
        with torch.no_grad():
            return (torch.softmax(m(worker.x), 1)[:, 1] - .5).abs().mean().item()

    def test_bad_teacher_pushes_forget_to_uniform_and_keeps_retain(self):
        student, _ = distill(self.original, self.original, self.retain, {**self.config, 'kd_mode': 'bad_teacher'}, 0, self.forget)
        self.assertLess(self.confidence(student, self.forget), .5 * self.confidence(self.original, self.forget))
        self.assertGreater(self.confidence(student, self.retain), .5 * self.confidence(self.original, self.retain))

    def test_full_local_with_identical_teacher_is_stationary(self):
        config = {**self.config, 'kd_mode': 'full_local', 'hard_weight': 0., 'kd_epochs': 1, 'kd_optimizer': 'sgd'}
        student, losses = distill(self.original, self.original, self.retain, config, 0, self.forget)
        self.assertLess(max(losses), 1e-6)
        self.assertTrue(params_close(student, self.original))

    def test_full_local_needs_forget_and_never_uses_hard_labels_on_f(self):
        with self.assertRaises(ValueError):
            distill(self.original, self.original, self.retain, {**self.config, 'kd_mode': 'full_local'}, 0)
        # Teacher = student and hard_weight = 1: R gives CE, F contributes zero, so loss < full-batch CE.
        config = {**self.config, 'kd_mode': 'full_local', 'hard_weight': 1., 'kd_epochs': 1, 'kd_lr': 0.,
                  'batch_size': 120}
        _, losses = distill(self.original, self.original, self.retain, config, 0, self.forget)
        retain_ce = mean_ce(self.original, self.retain.x, self.retain.y)
        self.assertAlmostEqual(losses[0], retain_ce * 80 / 120, places=5)

    def test_kd_stop_drops_forget_rows(self):
        config = {**self.config, 'kd_mode': 'bad_teacher', 'kd_epochs': 1, 'batch_size': 8}
        _, all_rows = distill(self.original, self.original, self.retain, config, 0, self.forget)
        _, stopped = distill(self.original, self.original, self.retain, config, 0, self.forget, stop_target=0.)
        order = np.random.default_rng(0).permutation(120)
        with_retain = sum((order[i:i+8] < 80).any() for i in range(0, 120, 8))
        self.assertEqual(len(all_rows), 15)
        self.assertEqual(len(stopped), with_retain)

    def test_unlearned_student_starts_from_teacher(self):
        teacher, _ = alternating(self.original, self.forget, self.retain, LEGACY, 1)
        config = {**self.config, 'kd_mode': 'bad_teacher', 'kd_student': 'unlearned', 'kd_lr': 0., 'kd_epochs': 1}
        student, _ = distill(self.original, teacher, self.retain, config, 0, self.forget)
        self.assertTrue(params_equal(student, teacher))
        self.assertTrue(all(p.requires_grad for p in student.parameters()))


class ControlAndMetricTests(unittest.TestCase):
    def test_retain_finetune_runs_exact_steps(self):
        x, y = toy(2)
        original = trained(x, y, 2, 20)
        retain = ClientWorker(0, x, y, np.arange(len(y)))
        config = {'retain_optimizer': 'adam', 'retain_lr': .01, 'batch_size': 50, 'clip': 1.}
        same, losses = retain_finetune(original, retain, 0, config, 0)
        self.assertEqual(losses, [])
        self.assertTrue(params_equal(same, original))
        _, losses = retain_finetune(original, retain, 7, config, 0)  # 3 batches/epoch: crosses epochs
        self.assertEqual(len(losses), 7)

    def test_noise_floor_leave_one_out(self):
        retrains = [np.zeros(4), np.full(4, .1), np.full(4, .3)]
        noise = retrain_noise(retrains)
        np.testing.assert_allclose(noise['loo_distance'], [.2, .15, .25])
        self.assertAlmostEqual(noise['sd'], .05)
        self.assertAlmostEqual(distance_to_retrains(np.full(4, .5), retrains), (.5 + .4 + .2) / 3)
        self.assertAlmostEqual(margin(noise, 2, .001), .1)
        self.assertEqual(margin(noise, 2, .5), .5)
        self.assertTrue(identifiable(.5, noise, 3))
        self.assertFalse(identifiable(.26, noise, 3))  # above max but within 3 sd of the mean
        with self.assertRaises(ValueError):
            retrain_noise(retrains[:2])

    def test_lira_separates_members_from_out_distribution(self):
        rng = np.random.default_rng(0)
        shadow = rng.normal(0, 1, (16, 2000))
        target = shadow.mean(0) + rng.normal(0, 1, 2000)
        target[:1000] += 2.5
        mu, sd = out_statistics(shadow)
        self.assertEqual(len(np.unique(sd)), 1)
        scores = lira_scores(target, mu, sd)
        y = np.tile([0, 1], 1000)
        out = attack_summary(y[:1000], scores[:1000], y[1000:], scores[1000:], [2., 2.])
        self.assertGreater(out['mia_auc'], .9)
        self.assertIsNotNone(out['mia_tpr_at_fpr_0_01'])
        same = attack_summary(y[:1000], scores[1000:], y[1000:], scores[1000:], [2., 2.])
        self.assertAlmostEqual(same['mia_auc'], .5)
        np.testing.assert_allclose(logit_confidence(np.array([1, 0]), np.array([.75, .75])), [np.log(3), -np.log(3)])
        with self.assertRaises(ValueError):
            out_statistics(shadow[:1])

    def test_near_duplicates_use_nonmember_distance_radius(self):
        train_x = np.array([[0.], [.05], [1.], [3.]])
        index, radius = near_duplicate_index(train_x, np.array([[2.], [2.2], [2.4]]), .5)
        self.assertAlmostEqual(radius, .8)  # median distance of 2.0/2.2/2.4 to the nearest train row
        np.testing.assert_array_equal(near_duplicates(index, np.array([[0.]]), radius), [0, 1])
        np.testing.assert_array_equal(near_duplicates(index, np.array([[1.9]]), .5), [])

    def test_decide_adds_pre_recovery_candidate_only_for_local_checks(self):
        from scripts.run_study import decide, summarize
        def entry(distance, f1=.9, fpr=.2, auc=.6):
            return {'distance_F': distance, 'utility_test': {'macro_f1': f1, 'fpr': fpr},
                    'lira_forget': {'mia_auc': auc}, 'lira_retained_control': {'mia_auc': .6}}
        metrics = {'original': entry(.05), 'retain_only': entry(.03), 'm/after_local': entry(.01, f1=.8),
                   'm/final': entry(.02), 'm/retain_matched': entry(.03), 'm/retain_matched_local': entry(.04)}
        noise = {'mean': .005, 'sd': .001, 'max': .007}
        gate = {'margin_sd_factor': 2, 'minimum_margin': 1e-4, 'identifiability_sd': 3, 'minimum_retained_lira_auc': .55,
                'minimum_macro_f1': .65, 'maximum_f1_drop': .03, 'maximum_fpr_increase': .02}
        final, pre = decide('m', {'stop_rule': 'proxy_calibrated'}, metrics, noise, .55, .9, gate, False)
        self.assertEqual((final['candidate'], final['matched_model']), ('final', 'm/retain_matched'))
        self.assertTrue(final['passed'])
        self.assertEqual((pre['candidate'], pre['matched_model']), ('pre_recovery', 'm/retain_matched_local'))
        self.assertAlmostEqual(pre['improvement_over_matched'], .03)
        self.assertFalse(pre['checks']['utility_preserved'])  # closer to retrain but F1 dropped 0.1
        self.assertFalse(pre['passed'])
        only = decide('m', {'stop_rule': 'proxy_calibrated', 'stop_check': 'after_recovery'}, metrics, noise, .55, .9, gate, False)
        self.assertEqual([d['candidate'] for d in only], ['final'])
        summary = summarize([final, pre], {'gate': {**gate, 'confidence': .95, 'minimum_valid_cases': 1,
                                                    'minimum_pass_rate': 1.}, 'methods': {'m': {}}}, False)
        self.assertEqual(set(summary), {'m@final', 'm@pre_recovery'})

    def test_mia_not_required_ignores_mia_checks(self):
        from scripts.run_study import decide
        entry = lambda distance, auc: {'distance_F': distance, 'utility_test': {'macro_f1': .9, 'fpr': .2},
                                       'lira_forget': {'mia_auc': auc}, 'lira_retained_control': {'mia_auc': .45}}
        metrics = {'original': entry(.05, .6), 'retain_only': entry(.03, .6), 'm/after_local': entry(.01, .9),
                   'm/final': entry(.01, .9), 'm/retain_matched': entry(.03, .6), 'm/retain_matched_local': entry(.04, .6)}
        noise = {'mean': .005, 'sd': .001, 'max': .007}
        gate = {'margin_sd_factor': 2, 'minimum_margin': 1e-4, 'identifiability_sd': 3, 'minimum_retained_lira_auc': .55,
                'minimum_macro_f1': .65, 'maximum_f1_drop': .03, 'maximum_fpr_increase': .02}
        strict = decide('m', {}, metrics, noise, .6, .9, gate, False)[0]
        self.assertFalse(strict['valid_case'])
        relaxed = decide('m', {}, metrics, noise, .6, .9, {**gate, 'mia_required': False}, False)[0]
        self.assertTrue(relaxed['valid_case'])
        self.assertFalse(relaxed['checks']['mia_positive_control'] or relaxed['checks']['mia_closer_to_retrain'])
        self.assertTrue(relaxed['passed'])

    def test_matched_indices_optional_class(self):
        member, nonmember = np.ones(10, dtype=int), np.tile([0, 1], 10)
        with self.assertRaises(ValueError):
            matched_indices(member, nonmember, 0)
        mi, ni = matched_indices(member, nonmember, 0, require_all_classes=False)
        self.assertEqual(len(mi), 10)
        self.assertTrue((nonmember[ni] == 1).all())


class ShadowPartitionTests(unittest.TestCase):
    def test_shadow_partition_retries_deterministically(self):
        from scripts.run_study import shadow_partition
        y = np.r_[np.zeros(10, dtype=int), np.ones(400, dtype=int)]
        parts, retries = shadow_partition(y, 0, 1., 5)
        again, same = shadow_partition(y, 0, 1., 5)
        self.assertGreater(retries, 0)
        self.assertEqual(retries, same)
        self.assertTrue(all(np.array_equal(a, b) for a, b in zip(parts, again)))
        self.assertTrue(all(set(y[p]) == {0, 1} for p in parts))
        with self.assertRaises(ValueError):
            shadow_partition(np.r_[np.zeros(2, dtype=int), np.ones(400, dtype=int)], 0, .1, 5)


class ScenarioTests(unittest.TestCase):
    def test_exclusive_family_partition_and_request(self):
        y = np.tile([0, 1], 2000)
        types = np.where(y == 0, 'normal', np.where(np.arange(4000) % 4 == 1, 'ransomware', 'dos'))
        clients = exclusive_family_partition(y, types, 'ransomware', 2, 42, alpha=5)
        np.testing.assert_array_equal(np.sort(np.concatenate(clients)), np.arange(4000))
        family = np.flatnonzero(types == 'ransomware')
        self.assertTrue(np.isin(family, clients[2]).all())
        self.assertFalse(any(np.isin(family, c).any() for i, c in enumerate(clients) if i != 2))
        f, r = family_request(types, clients[2], 'ransomware', 1., 1)
        np.testing.assert_array_equal(f, family)
        self.assertEqual(set(y[r]), {0, 1})
        half, _ = family_request(types, clients[2], 'ransomware', .5, 1)
        self.assertEqual(len(half), len(family) // 2)
        with self.assertRaises(ValueError):
            exclusive_family_partition(y, types, 'normal', 2, 42)

    def test_atypical_request_takes_unique_high_loss_rows(self):
        x = np.arange(40, dtype=np.float32).reshape(20, 2)
        x[1] = x[0]  # duplicate pair
        unique = unique_feature_rows(x)
        self.assertFalse(unique[0] or unique[1])
        self.assertEqual(unique.sum(), 18)
        y = np.tile([0, 1], 10)
        losses = np.linspace(0, 1, 20)
        losses[0] = losses[1] = 99  # highest loss but duplicated: never chosen
        f, r = atypical_request(y, losses, np.arange(20), .2, unique)
        np.testing.assert_array_equal(f, [16, 17, 18, 19])
        self.assertEqual(len(np.intersect1d(f, r)), 0)


class StudySmokeTest(unittest.TestCase):
    def test_synthetic_end_to_end_smoke_never_passes_gate(self):
        from scripts.make_synthetic_toniot import synthetic
        from src.data.prepare import prepare
        from scripts.run_study import run
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic().to_csv(tmp/'raw.csv', index=False)
            prepare(tmp/'raw.csv', Path('configs/data_connected.json'), tmp/'data')
            config = json.loads(Path('configs/study.json').read_text())
            config['methods'] = {k: config['methods'][k] for k in ['legacy_retain_kd', 'uniform_adam_bad_teacher']}
            config['smoke']['scenarios'] = config['smoke']['scenarios'][::2]  # random + family
            (tmp/'study.json').write_text(json.dumps(config))
            run(tmp/'data', tmp/'study.json', tmp/'run', smoke=True)
            gate = json.loads((tmp/'run'/'gate.json').read_text())
            coverage = pd.read_csv(tmp/'run'/'coverage.csv')
            self.assertTrue(gate['smoke_only'])
            self.assertFalse(gate['study_passed_any'])
            self.assertFalse(any(d['passed'] for d in gate['decisions']))
            self.assertEqual(set(coverage['status']), {'completed'})
            for case in (tmp/'run'/'cases').iterdir():
                metrics = json.loads((case/'metrics.json').read_text())
                self.assertEqual(metrics['noise_floor_F']['retrains'], 3)
                self.assertIn('uniform_adam_bad_teacher/retain_matched', metrics['models'])
                info = metrics['methods']['uniform_adam_bad_teacher']
                self.assertGreater(info['local_gradient_batches'] + info['kd_steps'], 0)

    def test_v3_smoke_proxy_recovery_check_and_near_exclusion(self):
        from scripts.make_synthetic_toniot import synthetic
        from src.data.prepare import prepare
        from scripts.run_study import run
        with tempfile.TemporaryDirectory() as tmp:
            tmp = Path(tmp)
            synthetic().to_csv(tmp/'raw.csv', index=False)
            prepare(tmp/'raw.csv', Path('configs/data_connected.json'), tmp/'data')
            config = json.loads(Path('configs/study_v3.json').read_text())
            config['methods'] = {k: config['methods'][k] for k in ['neggrad_plus_proxy', 'neggrad_plus_proxy_recovery']}
            config['lira'].update({'exclude': 'near', 'subset_fraction': 1.})
            config['smoke']['scenarios'] = config['smoke']['scenarios'][:1]
            (tmp/'study.json').write_text(json.dumps(config))
            run(tmp/'data', tmp/'study.json', tmp/'run', smoke=True)
            gate = json.loads((tmp/'run'/'gate.json').read_text())
            self.assertEqual(set(gate['methods']), {'neggrad_plus_proxy@final', 'neggrad_plus_proxy@pre_recovery',
                                                    'neggrad_plus_proxy_recovery@final'})
            case = next((tmp/'run'/'cases').iterdir())
            metrics = json.loads((case/'metrics.json').read_text())
            self.assertIn('proxy_calibrated', metrics['stop_references'])
            self.assertGreater(metrics['lira']['near_duplicates_excluded'], 0)
            info = metrics['methods']['neggrad_plus_proxy_recovery']
            self.assertEqual(info['stop_check'], 'after_recovery')
            self.assertGreaterEqual(info['recovery_rounds'], 1)


if __name__ == '__main__':
    unittest.main()


class OperatingPointAndCondenseTests(unittest.TestCase):
    def test_fpr_threshold_respects_budget_with_float32_ties(self):
        from src.eval.metrics import operating_threshold, utility
        y = np.r_[np.zeros(1000, dtype=int), np.ones(100, dtype=int)]
        p = np.r_[np.full(200, .7), np.linspace(0, .6, 800), np.full(100, .9)].astype(np.float32)
        threshold = operating_threshold(y, p, {'type': 'fpr', 'max_fpr': .05})
        self.assertLessEqual(utility(y, p, threshold)['fpr'], .05)  # 200 tied negatives must stay below
        self.assertEqual(utility(y, p, threshold)['recall_attack'], 1.)
        best = operating_threshold(y, p, {'type': 'max_f1'})
        self.assertGreaterEqual(utility(y, p, best)['macro_f1'], utility(y, p, .5)['macro_f1'])
        with self.assertRaises(ValueError):
            operating_threshold(y, p, {'type': 'unknown'})

    def test_detection_by_type_and_disagreement(self):
        from src.eval.metrics import detection_by_type
        from src.eval.noise_floor import disagreement_to_retrains
        out = detection_by_type(['normal', 'normal', 'dos', 'dos'], np.array([.1, .8, .9, .2], dtype=np.float32), .5)
        self.assertEqual(out['normal'], {'alert_rate': .5, 'rows': 2})
        self.assertEqual(out['dos']['alert_rate'], .5)
        self.assertAlmostEqual(disagreement_to_retrains(np.array([.9, .1]), [np.array([.9, .9]), np.array([.1, .1])]), .5)

    def test_condense_averages_repeats_and_uses_operating_point(self):
        from scripts.run_study import condense
        def entry(distance, f1, op_f1):
            return {'distance_F': distance, 'utility_test': {'macro_f1': f1, 'fpr': .3},
                    'operating_point': {'test': {'macro_f1': op_f1, 'fpr': .05}},
                    'lira_forget': None, 'lira_retained_control': None, 'forgotten_type_alert': .4}
        view = condense([entry(.01, .8, .9), entry(.03, .8, .7)], {'utility_source': 'operating_point'})
        self.assertAlmostEqual(view['distance_F'], .02)
        self.assertEqual(view['distance_F_repeats'], [.01, .03])
        self.assertAlmostEqual(view['utility_test']['macro_f1'], .8)
        self.assertEqual(view['utility_test']['fpr'], .05)
        self.assertTrue(np.isnan(view['lira_forget']['mia_auc']))
        self.assertEqual(condense([entry(.01, .8, .9)], {})['utility_test']['macro_f1'], .8)

    def test_relative_floor_and_report_checks(self):
        from scripts.run_study import decide
        def entry(distance, f1, alert):
            return {'distance_F': distance, 'utility_test': {'macro_f1': f1, 'fpr': .05},
                    'lira_forget': {'mia_auc': np.nan}, 'lira_retained_control': {'mia_auc': np.nan},
                    'forgotten_type_alert': alert}
        metrics = {'original': entry(.05, .60, .9), 'retain_only': entry(.03, .60, .5), 'm/after_local': entry(.01, .59, .9),
                   'm/final': entry(.01, .59, .95), 'm/retain_matched': entry(.03, .60, .5),
                   'm/retain_matched_local': entry(.04, .6, .5)}
        noise = {'mean': .005, 'sd': .001, 'max': .007}
        gate = {'margin_sd_factor': 2, 'minimum_margin': 1e-4, 'identifiability_sd': 3, 'minimum_retained_lira_auc': .55,
                'minimum_macro_f1': .65, 'maximum_f1_drop': .03, 'maximum_fpr_increase': .02, 'mia_required': False}
        absolute = decide('m', {}, metrics, noise, np.nan, .6, gate, False, retrain_alert=.2)[0]
        self.assertFalse(absolute['checks']['utility_floor'])
        relative = decide('m', {}, metrics, noise, np.nan, .6, {**gate, 'relative_floor': True}, False, retrain_alert=.2)[0]
        self.assertAlmostEqual(relative['utility_floor_used'], .57)
        self.assertFalse(relative['checks']['report_family_detection_closer_to_retrain'])
        self.assertTrue(relative['passed'])  # report_ checks never decide passing


class ReportStatisticsTests(unittest.TestCase):
    def test_holm_and_mean_ci(self):
        from scripts.report_study import holm, mean_ci, wilcoxon
        np.testing.assert_allclose(holm([.01, .04, .03, np.nan]), [.03, .06, .06, np.nan])
        mean, low, high = mean_ci([1., 2., 3.])
        self.assertEqual(mean, 2.)
        self.assertAlmostEqual(high - mean, stats_t_half_width(), places=6)
        self.assertTrue(np.isnan(wilcoxon([0., 0., 0.])))


def stats_t_half_width():
    from scipy import stats
    return stats.t.ppf(.975, 2) * 1. / np.sqrt(3)
