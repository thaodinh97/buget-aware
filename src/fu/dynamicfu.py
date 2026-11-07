"""DynamicFU (Wu et al., Sensors 26:3714, 2026, Sec. 4) adapted to sample-level deletion; `dynamicfu_inspired`.

Unit of deletion: a forget set F held by one client u, not the whole client. Shard s_u is retrained from
its original initialisation with u keeping R_u = D_u \\ F (src/fu/shard_retrain.py), so no gradient of
the output model touches F. The round budget follows the paper:
  contribution  Eq. (9)-(12) computed for F instead of a client:
                u_F = ||w+ - w*_{s_u}||_2, w+ = one local epoch on F from w*_{s_u};
                d_F = l_JS * JS(p_F || p_train) + l_pi * pi_F  (pi_F = attack share of F);
                q_F = |F1_val(W+) - F1_val(W*)|, W+ = W* with shard s_u replaced by w+;
                min-max normalised against the same raw scores of every client's full data D_i (the
                paper's candidate-client pool, computed once per trained ensemble), clipped to [0, 1];
                I_F = theta_u u + theta_d d + theta_q q.
  budget        Eq. (13)-(14), then the controller Eq. (16)-(21) with the three stopping rules of
                Sec. 4.4.3 and Drop measured against F1_val(W*).
I_F and W* depend on F, so the round count R_u carries information about F (at most log2(R_max) bits).
The returned shard is the exact retrain trajectory truncated at R_u; callers verify this bit for bit.

`dynamicfu_ffree` (ablation) removes both F-dependent inputs: eta_0 is a constant and Drop is measured
against the best validation F1 seen so far in this retraining. Everything else is identical.
"""
from dataclasses import dataclass
import numpy as np
from sklearn.metrics import f1_score
from src.fl.fedavg_core import state
from src.fl.sharded import fedavg_round, margins


@dataclass(frozen=True)
class ContributionConfig:
    theta_u: float
    theta_d: float
    theta_q: float
    lambda_js: float
    lambda_pi: float
    eps: float = 1e-12

    def __post_init__(self):
        weights = [self.theta_u, self.theta_d, self.theta_q, self.lambda_js, self.lambda_pi]
        if min(weights) < 0 or abs(sum(weights[:3]) - 1) > 1e-9 or abs(sum(weights[3:]) - 1) > 1e-9:
            raise ValueError('theta and lambda weights must be non-negative and each sum to 1')


@dataclass(frozen=True)
class ControllerConfig:
    rounds_base: int
    rounds_max: int
    eta_min: float
    eta_max: float
    rho: float
    mu: float
    omega_g: float
    omega_s: float
    tau_ctrl: float
    tau_drop: float
    eps: float = 1e-12

    def __post_init__(self):
        if self.rounds_base < 1 or self.rounds_max < 1:
            raise ValueError('Round budgets must be positive')
        if not 0 < self.eta_min <= self.eta_max:
            raise ValueError('Need 0 < eta_min <= eta_max')
        if min(self.rho, self.mu, self.omega_g, self.omega_s, self.tau_ctrl, self.tau_drop) < 0:
            raise ValueError('Controller weights and thresholds must be non-negative')
        if abs(self.omega_g + self.omega_s - 1) > 1e-9:
            raise ValueError('omega_g + omega_s must equal 1')


def binary_distribution(y):
    counts = np.bincount(np.asarray(y, dtype=np.int64), minlength=2).astype(np.float64)
    if counts.sum() == 0:
        raise ValueError('Empty label set')
    return counts / counts.sum()


def js_divergence(p, q):
    """Jensen-Shannon divergence in bits, so it lies in [0, 1]."""
    p, q = np.asarray(p, dtype=np.float64), np.asarray(q, dtype=np.float64)
    m = (p + q) / 2

    def kl(a, b):
        mask = a > 0
        return float(np.sum(a[mask] * np.log2(a[mask] / b[mask])))
    return (kl(p, m) + kl(q, m)) / 2


def ensemble_ce(z, y):
    return float(np.mean(np.logaddexp(0., z) - y * z))


def ensemble_f1(z, y):
    return float(f1_score(y, z >= 0, average='macro'))


class ContributionScorer:
    """Pool-normalised contribution I_F (Eq. 12 and the unified score of Sec. 4.2.4)."""

    def __init__(self, cfg, original_models, init_fn, fl, seed, train_distribution, validation):
        self.cfg, self.models, self.init_fn, self.fl, self.seed = cfg, original_models, init_fn, fl, seed
        self.train_distribution = train_distribution
        self.val_x, self.val_y = validation
        self.val_margins = [margins(m, self.val_x) for m in original_models]
        total = np.zeros(len(self.val_y))
        for z in self.val_margins:
            total = total + z
        self.f1_original = ensemble_f1(total, self.val_y)
        self.pool = None

    def raw(self, shard, worker, seed):
        """Raw (u, d, q) of the rows held by `worker` (F, or a client's D_i for the pool), Eq. (9)-(11).
        The short local update is one local epoch of the FL optimiser, run where the rows live."""
        updated, _, log = worker.train(self.models[shard], self.fl, seed)
        reference = state(self.models[shard])
        u = float(sum(((updated[k].double() - reference[k].double()) ** 2).sum() for k in reference)) ** .5
        y = worker.y.cpu().numpy()
        js = js_divergence(binary_distribution(y), self.train_distribution)
        pi = float((y == 1).mean())
        plus = self.init_fn(shard)
        plus.load_state_dict(updated)
        replaced = margins(plus, self.val_x)
        total = np.zeros(len(self.val_y))
        for s, z in enumerate(self.val_margins):
            total = total + (replaced if s == shard else z)
        q = abs(ensemble_f1(total, self.val_y) - self.f1_original)
        d = self.cfg.lambda_js * js + self.cfg.lambda_pi * pi
        return {'u': u, 'd': d, 'q': q, 'js': js, 'pi': pi, 'rows': int(len(y)), 'local_steps': log['steps']}

    def fit_pool(self, client_workers, shard_of):
        """Raw scores of every client's full data D_i: the normalisation pool."""
        self.pool = {c: self.raw(shard_of[c], w, self.seed + 7000 + 100 * c) for c, w in sorted(client_workers.items())}
        return self.pool

    def score(self, raw):
        if self.pool is None:
            raise ValueError('Fit the client pool before scoring requests')
        out = {}
        for key in ['u', 'd', 'q']:
            values = [p[key] for p in self.pool.values()]
            lo, hi = min(values), max(values)
            out[f'{key}_bar'] = float(np.clip((raw[key] - lo) / (hi - lo + self.cfg.eps), 0., 1.))
        out['I'] = self.cfg.theta_u * out['u_bar'] + self.cfg.theta_d * out['d_bar'] + self.cfg.theta_q * out['q_bar']
        return out


def budget(rounds_base, eta):
    """Eq. (14)/(21) with half-up rounding (Python's round() is banker's rounding)."""
    return max(1, int(np.floor(rounds_base * eta + .5)))


def initial_eta(contribution, cfg):
    """Eq. (13)."""
    return float(np.clip(cfg.eta_min + (cfg.eta_max - cfg.eta_min) * contribution, cfg.eta_min, cfg.eta_max))


class Controller:
    """Eq. (16)-(21). `drop_reference` = F1_val(W*) for DynamicFU, None for the best-so-far (F-free) rule."""

    def __init__(self, cfg, eta0, drop_reference=None):
        self.cfg, self.drop_reference = cfg, drop_reference
        self.eta = float(np.clip(eta0, cfg.eta_min, cfg.eta_max))
        self.best = None

    def start(self, f1_initial):
        self.best = f1_initial

    def step(self, r, loss_prev, loss, f1_prev, f1):
        c = self.cfg
        gain = max(0., loss_prev - loss) / (loss_prev + c.eps)          # Eq. (17)
        stab = abs(f1 - f1_prev)                                         # Eq. (18)
        ctrl = c.omega_g * gain + c.omega_s * stab
        reference = self.best if self.drop_reference is None else self.drop_reference
        drop = max(0., reference - f1)                                   # Eq. (19)
        self.eta = float(np.clip(self.eta + c.rho * ctrl - c.mu * drop, c.eta_min, c.eta_max))  # Eq. (20)
        target = budget(c.rounds_base, self.eta)                         # Eq. (21)
        if r >= target and ctrl <= c.tau_ctrl:
            reason = 'converged'
        elif drop >= c.tau_drop:
            reason = 'utility_drop'
        elif r >= c.rounds_max:
            reason = 'max_rounds'
        else:
            reason = None
        self.best = max(self.best, f1)
        return {'round': r, 'gain': gain, 'stab': stab, 'ctrl': ctrl, 'drop': drop, 'eta': self.eta,
                'target_rounds': target, 'stop_reason': reason}


def replay(losses, f1s, cfg, eta0, drop_reference=None):
    """Stopping decision from recorded per-round signals (index 0 = initialisation). The trajectory does
    not depend on the controller, so this equals running the controller online."""
    controller = Controller(cfg, eta0, drop_reference)
    controller.start(f1s[0])
    trace = []
    for r in range(1, min(cfg.rounds_max, len(losses) - 1) + 1):
        step = controller.step(r, losses[r - 1], losses[r], f1s[r - 1], f1s[r])
        trace.append(step)
        if step['stop_reason']:
            return r, trace
    raise ValueError('Signals end before the controller stops')


def make_signal_fn(original_models, shard, eval_set, validation):
    """(L_ret on the retained eval rows V, macro-F1 on validation) of the ensemble whose shard `shard` is
    the given model. Reads `original_models` only for the other shards, so it never sees F."""
    (ex, ey), (vx, vy) = eval_set, validation
    others_eval, others_val = np.zeros(len(ey)), np.zeros(len(vy))
    for s, m in enumerate(original_models):
        if s != shard:
            others_eval = others_eval + margins(m, ex)
            others_val = others_val + margins(m, vx)

    def signal(model):
        return ensemble_ce(others_eval + margins(model, ex), ey), ensemble_f1(others_val + margins(model, vx), vy)
    return signal


def controlled_retrain(init_model, workers, fl, seed, cfg, eta0, drop_reference, signal_fn):
    """Retrain the affected shard round by round until the controller stops (Algorithm 1, lines 8-19).
    `signal_fn(model) -> (L_ret, F1_val)` must only read retained data, validation and untouched shards."""
    if not workers:
        raise ValueError('The affected shard has no client left after the request')
    controller = Controller(cfg, eta0, drop_reference)
    current = init_model
    loss_prev, f1_prev = signal_fn(current)
    controller.start(f1_prev)
    trace = [{'round': 0, 'loss_ret': loss_prev, 'f1_val': f1_prev, 'eta': controller.eta,
              'target_rounds': budget(cfg.rounds_base, controller.eta), 'stop_reason': None}]
    logs = []
    for r in range(1, cfg.rounds_max + 1):
        current, log = fedavg_round(current, workers, fl, r - 1, seed)
        logs.append(log)
        loss, f1 = signal_fn(current)
        step = controller.step(r, loss_prev, loss, f1_prev, f1)
        trace.append({'loss_ret': loss, 'f1_val': f1, **step})
        loss_prev, f1_prev = loss, f1
        if step['stop_reason']:
            break
    return current, logs, trace
