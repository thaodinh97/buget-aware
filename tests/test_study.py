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
from src.eval.lira import logit_confidence, out_statistics, lira_scores, attack_summary
from src.eval.metrics import matched_indices
from src.eval.noise_floor import retrain_noise, distance_to_retrains, margin, identifiable
from src.fl.fedavg_core import ClientWorker, batches, model
from src.fu.baselines import retain_finetune
from src.fu.conpfu_inspired import alternating, distill, forget_loss, forget_step, gradient_batches, mean_ce

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

    def test_matched_indices_optional_class(self):
        member, nonmember = np.ones(10, dtype=int), np.tile([0, 1], 10)
        with self.assertRaises(ValueError):
            matched_indices(member, nonmember, 0)
        mi, ni = matched_indices(member, nonmember, 0, require_all_classes=False)
        self.assertEqual(len(mi), 10)
        self.assertTrue((nonmember[ni] == 1).all())


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


if __name__ == '__main__':
    unittest.main()
