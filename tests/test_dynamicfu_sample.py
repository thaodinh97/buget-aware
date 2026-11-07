import copy
import unittest
import numpy as np
import torch
from scipy.special import softmax
from src.data.federated import partition, partition_min_size, forget_request
from src.fl.fedavg_core import ClientWorker, fedavg, state
from src.fl.sharded import (assign_shards, shard_of_client, init_shard, shard_workers, train_sharded,
                            fedavg_round, margins, ensemble_probability, states_equal, state_hash)
from src.fu.shard_retrain import affected_shards, retrain_trajectory
from src.fu.dynamicfu import (ContributionConfig, ContributionScorer, ControllerConfig, Controller, budget,
                              initial_eta, js_divergence, binary_distribution, replay, make_signal_fn,
                              controlled_retrain)

DIM, HIDDEN, SEED = 5, [8, 4], 7
FL = {'local_epochs': 1, 'batch_size': 32, 'lr': 1e-2}
ROUNDS = 5


def synthetic(n=900, seed=0):
    rng = np.random.default_rng(seed)
    x = rng.normal(size=(n, DIM)).astype(np.float32)
    y = (x[:, 0] + .5 * x[:, 1] + .3 * rng.normal(size=n) > 0).astype(np.int64)
    return x, y


def ctrl(**overrides):
    raw = dict(rounds_base=ROUNDS, rounds_max=ROUNDS, eta_min=.2, eta_max=1., rho=.5, mu=1., omega_g=.5,
               omega_s=.5, tau_ctrl=.01, tau_drop=.05)
    raw.update(overrides)
    return ControllerConfig(**raw)


class Federation:
    """6 clients in 3 shards; a stratified sample request on the largest client."""
    def __init__(self, x, y, fraction=.5):
        self.x, self.y = x, y
        self.parts, _ = partition_min_size(y, SEED, 1., 6, 40)
        self.shards = assign_shards(6, 3, SEED)
        self.target = int(np.argmax([len(p) for p in self.parts]))
        self.f, self.r = forget_request(y, self.parts[self.target], fraction, SEED + 1)
        self.shard = affected_shards(self.shards, [self.target])[0]
        self.val = synthetic(300, seed=99)

    def workers(self, retained=False):
        return {c: ClientWorker(c, self.x[idx], self.y[idx], idx, device='cpu')
                for c, idx in ((c, self.r if retained and c == self.target else p) for c, p in enumerate(self.parts))}

    def init_fn(self, s):
        return init_shard(DIM, HIDDEN, SEED, s)

    def unlearn(self, cfg):
        original, _ = train_sharded(DIM, HIDDEN, self.shards, self.workers(), FL, ROUNDS, SEED)
        retained = shard_workers(self.shards[self.shard], self.workers(True))
        ev = np.concatenate([w.row_ids for w in retained])[:200]
        signal = make_signal_fn(original, self.shard, (self.x[ev], self.y[ev]), self.val)
        traj, _ = retrain_trajectory(self.init_fn(self.shard), retained, FL, ROUNDS, SEED)
        scorer = ContributionScorer(ContributionConfig(1 / 3, 1 / 3, 1 / 3, .5, .5), original, self.init_fn, FL, SEED,
                                    binary_distribution(self.y), self.val)
        scorer.fit_pool(self.workers(), shard_of_client(self.shards))
        forget = ClientWorker(self.target, self.x[self.f], self.y[self.f], self.f, device='cpu')
        score = scorer.score(scorer.raw(self.shard, forget, SEED + 3))
        out = {'original': original, 'trajectory': traj, 'score': score}
        for arm, eta0, ref in [('dynamicfu', initial_eta(score['I'], cfg), scorer.f1_original),
                               ('ffree', .6, None)]:
            model, logs, trace = controlled_retrain(self.init_fn(self.shard), retained, FL, SEED, cfg, eta0, ref, signal)
            out[arm] = (state(model), len(logs), trace, eta0, ref)
        return out


class DynamicFUSampleTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        torch.use_deterministic_algorithms(True)

    def test_partition_min_size(self):
        y = np.r_[np.zeros(300, int), np.ones(1200, int)]
        parts, attempts = partition_min_size(y, 3, .5, 10, 30)
        self.assertGreaterEqual(attempts, 1)
        self.assertTrue(all(len(p) >= 30 for p in parts))
        np.testing.assert_array_equal(np.sort(np.concatenate(parts)), np.arange(len(y)))
        for a, b in zip(parts, partition_min_size(y, 3, .5, 10, 30)[0]):
            np.testing.assert_array_equal(a, b)
        z = np.tile([0, 1], 10000)
        first, attempts = partition_min_size(z, 42, .5, 5, 1)
        self.assertEqual(attempts, 1)
        for a, b in zip(first, partition(z, 42)):
            np.testing.assert_array_equal(a, b)
        with self.assertRaises(ValueError):
            partition_min_size(y, 3, .5, 10, 200)

    def test_assign_shards(self):
        shards = assign_shards(20, 5, 42)
        np.testing.assert_array_equal(np.sort(np.concatenate(shards)), np.arange(20))
        self.assertEqual({len(s) for s in shards}, {4})
        self.assertEqual([s.tolist() for s in shards], [s.tolist() for s in assign_shards(20, 5, 42)])
        with self.assertRaises(ValueError):
            assign_shards(3, 4, 0)
        with self.assertRaises(ValueError):
            affected_shards(assign_shards(6, 3, 0), [6])

    def test_ensemble_is_softmax_of_summed_logits(self):
        x, _ = synthetic(50)
        models = [init_shard(DIM, HIDDEN, SEED, s) for s in range(3)]
        with torch.no_grad():
            logits = sum(m(torch.as_tensor(x)).double() for m in models).numpy()
        np.testing.assert_allclose(ensemble_probability([margins(m, x) for m in models]), softmax(logits, 1)[:, 1], rtol=1e-6)

    def test_trajectory_is_full_retrain_of_affected_shard(self):
        fed = Federation(*synthetic())
        original, _ = train_sharded(DIM, HIDDEN, fed.shards, fed.workers(), FL, ROUNDS, SEED)
        retained = fed.workers(True)
        self.assertFalse(set(fed.f.tolist()) & set(np.concatenate([w.row_ids for w in retained.values()]).tolist()))
        full, _ = train_sharded(DIM, HIDDEN, fed.shards, retained, FL, ROUNDS, SEED)
        traj, logs = retrain_trajectory(fed.init_fn(fed.shard), shard_workers(fed.shards[fed.shard], retained), FL, ROUNDS, SEED)
        self.assertEqual(len(traj), ROUNDS + 1)
        self.assertTrue(states_equal(traj[-1], state(full[fed.shard])))
        self.assertTrue(states_equal(traj[0], state(fed.init_fn(fed.shard))))
        for s in range(3):
            if s != fed.shard:
                self.assertTrue(states_equal(state(full[s]), state(original[s])))
        whole, _ = fedavg(fed.init_fn(0), shard_workers(fed.shards[0], retained), FL, 3, SEED)
        current = fed.init_fn(0)
        for r in range(3):
            current, _ = fedavg_round(current, shard_workers(fed.shards[0], retained), FL, r, SEED)
        self.assertTrue(states_equal(state(whole), state(current)))

    def test_js_and_contribution_normalisation(self):
        self.assertAlmostEqual(js_divergence([.3, .7], [.3, .7]), 0.)
        self.assertAlmostEqual(js_divergence([1, 0], [0, 1]), 1.)
        self.assertAlmostEqual(js_divergence([.2, .8], [.6, .4]), js_divergence([.6, .4], [.2, .8]))
        fed = Federation(*synthetic())
        original, _ = train_sharded(DIM, HIDDEN, fed.shards, fed.workers(), FL, ROUNDS, SEED)
        scorer = ContributionScorer(ContributionConfig(1 / 3, 1 / 3, 1 / 3, .5, .5), original, fed.init_fn, FL, SEED,
                                    binary_distribution(fed.y), fed.val)
        with self.assertRaises(ValueError):
            scorer.score({'u': 1., 'd': 1., 'q': 1.})
        pool = scorer.fit_pool(fed.workers(), shard_of_client(fed.shards))
        top = max(pool, key=lambda c: pool[c]['u'])
        self.assertAlmostEqual(scorer.score(pool[top])['u_bar'], 1., places=6)
        low = scorer.score({'u': -1., 'd': -1., 'q': -1.})
        self.assertEqual((low['u_bar'], low['d_bar'], low['q_bar'], low['I']), (0., 0., 0., 0.))
        self.assertEqual(scorer.score({'u': 1e9, 'd': 1e9, 'q': 1e9})['I'], 1.)
        with self.assertRaises(ValueError):
            ContributionConfig(.5, .5, .5, .5, .5)

    def test_controller_rules(self):
        cfg = ctrl(rounds_base=10, rounds_max=10, tau_drop=.02)
        c = Controller(cfg, .5, drop_reference=.8)
        c.start(.1)
        step = c.step(1, 1., 1., .7, .7)                           # 0.1 below W*: paper's rule stops
        self.assertEqual(step['stop_reason'], 'utility_drop')
        c = Controller(cfg, .5, drop_reference=None)                # best-so-far reference
        c.start(.1)
        self.assertIsNone(c.step(1, 1., .9, .1, .7)['stop_reason'])
        self.assertEqual(c.step(2, .9, .9, .7, .65)['stop_reason'], 'utility_drop')
        c = Controller(cfg, .5, drop_reference=.8)
        c.start(.8)
        self.assertIsNone(c.step(4, 1., 1., .8, .8)['stop_reason'])  # converged signal but r < target 5
        self.assertEqual(c.step(5, 1., 1., .8, .8)['stop_reason'], 'converged')
        c = Controller(cfg, .5, drop_reference=.8)
        c.start(.8)
        self.assertGreater(c.step(4, 1., .1, .8, .8)['target_rounds'], 5)
        self.assertEqual(c.step(10, 1., .5, .8, .8)['stop_reason'], 'max_rounds')
        self.assertEqual(budget(15, .5), 8)
        self.assertEqual(initial_eta(0., cfg), .2)
        self.assertEqual(initial_eta(1., cfg), 1.)
        with self.assertRaises(ValueError):
            ctrl(omega_g=.7)

    def test_arms_are_truncated_trajectories_and_replay_matches(self):
        fed = Federation(*synthetic())
        out = fed.unlearn(ctrl())
        traj = out['trajectory']
        losses, f1s = [], []
        retained = shard_workers(fed.shards[fed.shard], fed.workers(True))
        ev = np.concatenate([w.row_ids for w in retained])[:200]
        signal = make_signal_fn(out['original'], fed.shard, (fed.x[ev], fed.y[ev]), fed.val)
        for st in traj:
            m = fed.init_fn(fed.shard)
            m.load_state_dict(st)
            loss, f1 = signal(m)
            losses.append(loss)
            f1s.append(f1)
        for arm in ['dynamicfu', 'ffree']:
            arm_state, rounds, trace, eta0, ref = out[arm]
            self.assertTrue(states_equal(arm_state, traj[rounds]), arm)
            stop, replayed = replay(losses, f1s, ctrl(), eta0, ref)
            self.assertEqual(stop, rounds)
            self.assertEqual([t['stop_reason'] for t in trace[1:]], [t['stop_reason'] for t in replayed])
            self.assertEqual([t['loss_ret'] for t in trace], losses[:rounds + 1])

    def test_ffree_independent_of_forgotten_content_dynamicfu_only_through_rounds(self):
        x, y = synthetic()
        fed_a = Federation(x, y)
        x2, y2 = x.copy(), y.copy()
        rng = np.random.default_rng(5)
        x2[fed_a.f] = rng.normal(3., 2., size=(len(fed_a.f), DIM)).astype(np.float32)
        y2[fed_a.f] = 1 - y2[fed_a.f]
        fed_b = copy.copy(fed_a)
        fed_b.x, fed_b.y = x2, y2
        a, b = fed_a.unlearn(ctrl()), fed_b.unlearn(ctrl())
        self.assertFalse(states_equal(state(a['original'][fed_a.shard]), state(b['original'][fed_a.shard])))
        self.assertEqual([state_hash(s) for s in a['trajectory']], [state_hash(s) for s in b['trajectory']])
        self.assertEqual(state_hash(a['ffree'][0]), state_hash(b['ffree'][0]))
        self.assertEqual(a['ffree'][2], b['ffree'][2])
        self.assertNotEqual(a['score'], b['score'])                 # I_F reads F ...
        for out in (a, b):                                          # ... but only selects a round
            self.assertTrue(states_equal(out['dynamicfu'][0], a['trajectory'][out['dynamicfu'][1]]))


if __name__ == '__main__':
    unittest.main()
