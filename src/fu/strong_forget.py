"""Stronger local forget operators with Adam and a forget-loss ceiling.

ga_adam:      maximise CE(F).
neggrad_plus: minimise alpha*CE(R) - (1-alpha)*CE(F) on paired batches.
Both stop early once the forget-batch CE exceeds the starting CE on all of F
plus `ce_margin` nats. The guard is relative because a partly learned F (e.g.
a weak backdoor) can already have a high absolute CE. The realised step count
is logged.
"""
import copy
import torch
from torch import nn
from src.fu.step_budget import full_batches


def _forget(original, forget, retain, cfg, steps, seed, alpha):
    m = copy.deepcopy(original).train()
    opt = torch.optim.Adam(m.parameters(), lr=cfg['lr'])
    fs = full_batches(len(forget.y), cfg['batch_size'], seed + 1)
    rs = full_batches(len(retain.y), cfg['batch_size'], seed + 2) if retain is not None else None
    with torch.no_grad():
        ceiling = nn.functional.cross_entropy(m(forget.x), forget.y).item() + cfg['ce_margin']
    logs = []
    for step in range(steps):
        idx = next(fs); opt.zero_grad()
        ce_f = nn.functional.cross_entropy(m(forget.x[idx]), forget.y[idx])
        if ce_f.item() > ceiling:
            logs.append({'step': step, 'stopped': 'ce_ceiling', 'forget_ce': ce_f.item()}); break
        loss = -ce_f
        record = {'step': step, 'forget_ce': ce_f.item()}
        if rs is not None:
            j = next(rs); ce_r = nn.functional.cross_entropy(m(retain.x[j]), retain.y[j])
            loss = alpha*ce_r - (1 - alpha)*ce_f; record['retain_ce'] = ce_r.item()
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite forget loss')
        loss.backward()
        record['gradient_norm'] = nn.utils.clip_grad_norm_(m.parameters(), cfg['clip'], error_if_nonfinite=True).item()
        opt.step(); logs.append({**record, 'optimizer_steps': 1})
    return m, logs


def ga_adam(original, forget, cfg, steps, seed):
    return _forget(original, forget, None, cfg, steps, seed, None)


def neggrad_plus(original, forget, retain, cfg, steps, seed):
    return _forget(original, forget, retain, cfg, steps, seed, cfg['alpha'])
