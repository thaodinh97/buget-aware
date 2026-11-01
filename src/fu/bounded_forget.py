"""Local forget operators with bounded objectives, a target-level stop and compute accounting.

Forget terms on a batch of F (all minimised):
  ga:           -CE(F, y)                                unbounded; alpha > 0 gives NegGrad+
  random_label: CE(F, labels drawn uniformly per step)   >= 0; for two classes its expectation is `uniform`
  uniform:      KL(U || p) on F                          >= 0; zero iff p = 0.5 on every row
  soft_target:  KL(t || p) on F for per-row targets t    >= 0; e.g. the neighbour reference
loss = alpha*CE(R) + (1-alpha)*forget_term; alpha = 0 draws no retain batch.
The optimizer is built once per call, so Adam moments persist across steps.
cfg['projection'] = {'threshold', 'max_rows'} restricts every step to the
retain-orthogonal complement (src/fu/projection.py), with bases from the retain worker.

Stop rules, checked on all of F before every `check_every`-th step:
  target:     CE(F) >= target_ce (the runner supplies a nonmember, reference or retrain-cloud level);
              with cfg['stop_statistic'] = 'prob', instead the mean true-label probability on F <= target,
              which is less sensitive than mean CE to how confident the reference is on its tail;
  ce_ceiling: CE(F) > starting CE(F) + ce_margin (divergence guard, as in strong_forget).
Compute: backward_samples counts rows passed forward+backward through the model;
check_samples counts forward-only rows spent on the stop rule.
"""
import copy
import math
import numpy as np
import torch
from torch import nn
from src.fu.projection import retain_bases, project_gradients, project_update, snapshot
from src.fu.step_budget import full_batches

OBJECTIVES = ('ga', 'random_label', 'uniform', 'soft_target')


def forget_ce(m, worker):
    with torch.no_grad(): return nn.functional.cross_entropy(m(worker.x), worker.y).item()


def forget_prob(m, worker):
    """Mean softmax probability of the true label over the worker's rows."""
    with torch.no_grad(): return torch.softmax(m(worker.x), 1).gather(1, worker.y[:, None]).mean().item()


def uniform_kl(logits):
    """Mean KL(U || softmax(logits)); 0 iff every row predicts the uniform distribution."""
    return (-nn.functional.log_softmax(logits, 1).mean(1) - math.log(logits.shape[1])).mean()


def forget_term(objective, logits, y, rng, targets=None):
    if objective == 'ga': return -nn.functional.cross_entropy(logits, y)
    if objective == 'soft_target':
        if targets is None: raise ValueError('soft_target needs per-row targets')
        return (targets*(torch.log(targets.clamp_min(1e-12)) - nn.functional.log_softmax(logits, 1))).sum(1).mean()
    if objective == 'random_label':
        labels = torch.as_tensor(rng.integers(0, logits.shape[1], len(y)), device=y.device)
        return nn.functional.cross_entropy(logits, labels)
    if objective == 'uniform': return uniform_kl(logits)
    raise ValueError(f'Unknown forget objective: {objective}')


def make_optimizer(params, cfg):
    name = cfg.get('optimizer', 'adam')
    if name == 'adam': return torch.optim.Adam(params, lr=cfg['lr'])
    if name == 'sgd': return torch.optim.SGD(params, lr=cfg['lr'])
    raise ValueError(f'Unknown optimizer: {name}')


def clipped_step(m, opt, loss, clip, message, bases=None):
    if not torch.isfinite(loss): raise FloatingPointError(message)
    loss.backward()
    if bases is not None: project_gradients(m, bases); before = snapshot(m)
    norm = nn.utils.clip_grad_norm_(m.parameters(), clip, error_if_nonfinite=True).item()
    opt.step()
    if bases is not None: project_update(m, before, bases)
    return norm


class StopRule:
    """Target/ceiling stop over all of F; inactive (and free) when both are None. The ceiling always uses CE."""
    def __init__(self, m, forget, target_ce=None, ce_margin=None, check_every=1, statistic='ce'):
        if check_every < 1: raise ValueError('check_every must be positive')
        if statistic not in ('ce', 'prob'): raise ValueError(f'Unknown stop statistic: {statistic}')
        self.forget, self.target, self.every, self.statistic = forget, target_ce, check_every, statistic
        self.active = target_ce is not None or ce_margin is not None
        self.start = forget_ce(m, forget) if self.active else None
        self.start_value = (self.start if statistic == 'ce' else forget_prob(m, forget)) if self.active else None
        self.check_samples = len(forget.y) if self.active else 0
        self.ceiling = None if ce_margin is None else self.start + ce_margin

    def __call__(self, m, step):
        if not self.active or step % self.every: return None
        ce, value = self.start, self.start_value
        if step:
            ce = forget_ce(m, self.forget); self.check_samples += len(self.forget.y)
            value = ce if self.statistic == 'ce' else forget_prob(m, self.forget)
        if self.target is not None and (value >= self.target if self.statistic == 'ce' else value <= self.target): return 'target'
        if self.ceiling is not None and ce > self.ceiling: return 'ce_ceiling'
        return None


def local_forget(original, forget, retain, cfg, seed, target_ce=None, soft_targets=None):
    """cfg: objective, alpha, optimizer, lr, batch_size, clip, max_steps, check_every, ce_margin, projection."""
    if cfg['objective'] not in OBJECTIVES: raise ValueError(f"Unknown forget objective: {cfg['objective']}")
    alpha = cfg.get('alpha', 0.)
    if not 0 <= alpha < 1: raise ValueError('alpha must be in [0, 1)')
    if alpha and retain is None: raise ValueError('alpha > 0 needs a retain worker')
    proj = cfg.get('projection')
    if proj and retain is None: raise ValueError('projection needs a retain worker')
    m = copy.deepcopy(original).train(); opt = make_optimizer(m.parameters(), cfg)
    bases = retain_bases(m, retain, proj['threshold'], proj['max_rows'], seed + 5) if proj else None
    fs = full_batches(len(forget.y), cfg['batch_size'], seed + 1)
    rs = full_batches(len(retain.y), cfg['batch_size'], seed + 2) if alpha else None
    rng = np.random.default_rng(seed + 3)
    stop = StopRule(m, forget, target_ce, cfg.get('ce_margin'), cfg.get('check_every', 1), cfg.get('stop_statistic', 'ce'))
    logs, backward, reason = [], 0, 'max_steps'
    for step in range(cfg['max_steps']):
        hit = stop(m, step)
        if hit: reason = hit; break
        idx = next(fs); opt.zero_grad()
        term = forget_term(cfg['objective'], m(forget.x[idx]), forget.y[idx], rng, None if soft_targets is None else soft_targets[idx])
        record, loss = {'step': step, 'forget_term': term.item()}, term
        backward += len(idx)
        if alpha:
            j = next(rs); ce_r = nn.functional.cross_entropy(m(retain.x[j]), retain.y[j])
            loss = alpha*ce_r + (1 - alpha)*term; record['retain_ce'] = ce_r.item(); backward += len(j)
        record['gradient_norm'] = clipped_step(m, opt, loss, cfg['clip'], 'Nonfinite forget loss', bases)
        logs.append({**record, 'optimizer_steps': 1})
    info = {'steps': len(logs), 'stop_reason': reason, 'stop_target_ce': target_ce,
            'stop_start_ce': stop.start, 'stop_end_ce': forget_ce(m, forget),
            'backward_samples': backward, 'check_samples': stop.check_samples}
    if bases is not None: info['projection_ranks'] = '/'.join(str(b.shape[1]) for b in bases)
    if stop.statistic == 'prob': info.update({'stop_statistic': 'prob', 'stop_end_prob': forget_prob(m, forget)})
    return m, logs, info


def matched_steps(backward_samples, batch_size):
    """Retain-only steps that spend the same forward+backward rows as an FU arm."""
    return math.ceil(backward_samples / batch_size)


def retain_matched(original, retain, cfg, steps, seed):
    """Compute-matched retain-only control: the optimizer of cfg on CE(R) for `steps` steps."""
    m = copy.deepcopy(original).train(); opt = make_optimizer(m.parameters(), cfg)
    rs = full_batches(len(retain.y), cfg['batch_size'], seed + 2); logs = []
    for step in range(steps):
        idx = next(rs); opt.zero_grad()
        loss = nn.functional.cross_entropy(m(retain.x[idx]), retain.y[idx])
        norm = clipped_step(m, opt, loss, cfg['clip'], 'Nonfinite retained loss')
        logs.append({'step': step, 'retain_ce': loss.item(), 'gradient_norm': norm, 'optimizer_steps': 1})
    return m, logs


def retain_matched_many(original, retain, cfg, steps, seed):
    """retain_matched for several budgets in one pass: {k: model after k steps}. Identical to separate
    calls, which replay the same batch stream and optimizer state up to step k; k = 0 is a copy."""
    wanted = sorted(set(int(k) for k in steps)); out = {}
    m = copy.deepcopy(original).train(); opt = make_optimizer(m.parameters(), cfg)
    rs = full_batches(len(retain.y), cfg['batch_size'], seed + 2)
    if wanted and wanted[0] == 0: out[0] = copy.deepcopy(m)
    for step in range(1, (wanted[-1] if wanted else 0) + 1):
        idx = next(rs); opt.zero_grad()
        clipped_step(m, opt, nn.functional.cross_entropy(m(retain.x[idx]), retain.y[idx]), cfg['clip'], 'Nonfinite retained loss')
        if step in out or step in wanted: out[step] = copy.deepcopy(m)
    return out


def class_proportional(member_y, pool_y, cap, seed):
    """Pool indices with the class mix of member_y, min(cap, |members|) rows, no replacement."""
    member_y, pool_y = np.asarray(member_y), np.asarray(pool_y); n = min(cap, len(member_y))
    classes, counts = np.unique(member_y, return_counts=True)
    want = np.floor(counts/counts.sum()*n).astype(int); want[np.argmax(counts)] += n - want.sum()
    rng = np.random.default_rng(seed); picked = []
    for c, k in zip(classes, want):
        pool = np.flatnonzero(pool_y == c)
        if len(pool) < k: raise ValueError(f'Nonmember pool has {len(pool)} rows of class {c}, needs {k}')
        picked.append(rng.choice(pool, k, replace=False))
    return np.sort(np.concatenate(picked))
