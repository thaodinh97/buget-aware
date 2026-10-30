"""Global-start alternating forget/retain updates and client-local KD adaptations.

Defaults (SGD, unbounded -CE, fixed cycles, retain-only KD) reproduce the locked pilot.
Other options are study adaptations: bounded forget objectives, Adam, NegGrad+, a loss-based
stopping rule and KD modes that carry a forgetting signal on F. F is only read by the target
client's local operation; recovery never sees it.
"""
import copy
import torch
from torch import nn
from src.fl.fedavg_core import batches

FORGET_OBJECTIVES = ('neg_ce', 'neggrad_plus', 'random_label', 'uniform')
STOP_RULES = ('none', 'nonmember', 'oracle_retrain')
KD_MODES = ('retain_only', 'full_local', 'bad_teacher')


def make_optimizer(parameters, name, lr):
    if name == 'sgd':
        return torch.optim.SGD(parameters, lr=lr)
    if name == 'adam':
        return torch.optim.Adam(parameters, lr=lr)
    raise ValueError(f'Unknown optimizer: {name}')


def mean_ce(model, x, y, batch_size=4096):
    with torch.no_grad():
        total = sum(nn.functional.cross_entropy(model(x[i:i+batch_size]), y[i:i+batch_size], reduction='sum').item()
                    for i in range(0, len(y), batch_size))
    return total / len(y)


def forget_loss(logits, y, objective, generator=None):
    """Loss minimised on F. Only neg_ce/neggrad_plus are unbounded below."""
    if objective in ('neg_ce', 'neggrad_plus'):
        return -nn.functional.cross_entropy(logits, y)
    if objective == 'random_label':
        # Random incorrect label; for two classes this is the flipped label.
        offset = torch.randint(1, logits.shape[1], y.shape, generator=generator)
        return nn.functional.cross_entropy(logits, (y + offset) % logits.shape[1])
    if objective == 'uniform':
        # CE to the uniform distribution; minimum log(classes) at p = 1/classes.
        return -nn.functional.log_softmax(logits, dim=1).mean(dim=1).mean()
    raise ValueError(f'Unknown forget objective: {objective}')


def forget_step(model, x, y, anchor, lr, regularization, clip, optimizer=None, objective='neg_ce',
                retain=None, retain_weight=.5, generator=None):
    optimizer = optimizer or torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    logits = model(x)
    ce = nn.functional.cross_entropy(logits, y)
    penalty = sum((p - a).square().sum() for p, a in zip(model.parameters(), anchor))
    loss = forget_loss(logits, y, objective, generator)
    if objective == 'neggrad_plus':
        if retain is None:
            raise ValueError('NegGrad+ needs a retain batch')
        loss = (1 - retain_weight) * loss + retain_weight * nn.functional.cross_entropy(model(retain[0]), retain[1])
    loss = loss + regularization * penalty
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite FU loss')
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    optimizer.step()
    return {'forget_ce': ce.item(), 'loss': loss.item(), 'gradient_norm_before_clip': norm.item(),
            'batches': 2 if objective == 'neggrad_plus' else 1}


def _retain_batches(retain, batch_size, seed):
    epoch = 0
    while True:
        yield from batches(len(retain.y), batch_size, seed + epoch)
        epoch += 1


def alternating(original, forget, retain, config, seed, stop_target=None):
    """Forget phase on F, then retain phase on R, for at most config['cycles'] cycles.

    With a stop rule the forget phase ends once mean CE on F reaches stop_target, and cycling
    ends once CE on F is still at the target after the retain repair phase.
    """
    objective = config.get('forget_objective', 'neg_ce')
    stop_rule = config.get('stop_rule', 'none')
    if objective not in FORGET_OBJECTIVES or stop_rule not in STOP_RULES:
        raise ValueError('Unknown forget objective or stop rule')
    if (stop_rule == 'none') != (stop_target is None):
        raise ValueError('stop_target must be given exactly when a stop rule is active')
    local = copy.deepcopy(original)
    local.train()
    anchor = [p.detach().clone() for p in local.parameters()]
    # SGD has no state, so persistent optimizers match the pilot's per-step/per-cycle SGD exactly.
    forget_optimizer = make_optimizer(local.parameters(), config.get('forget_optimizer', 'sgd'), config['forget_lr'])
    retain_optimizer = make_optimizer(local.parameters(), config.get('retain_optimizer', 'sgd'), config['retain_lr'])
    generator = torch.Generator().manual_seed(seed)
    paired = _retain_batches(retain, config['batch_size'], seed + 500000)
    retain_steps = config.get('retain_steps_per_cycle')
    logs, reached = [], stop_target is None
    for cycle in range(config['cycles']):
        reached_in_forget = False
        for idx in batches(len(forget.y), config['batch_size'], seed + cycle):
            if stop_target is not None and mean_ce(local, forget.x, forget.y) >= stop_target:
                reached_in_forget = True
                break
            extra = None
            if objective == 'neggrad_plus':
                r = next(paired)
                extra = (retain.x[r], retain.y[r])
            logs.append({'cycle': cycle, 'phase': 'forget', **forget_step(
                local, forget.x[idx], forget.y[idx], anchor, config['forget_lr'], config['regularization'],
                config['clip'], forget_optimizer, objective, extra, config.get('neggrad_retain_weight', .5), generator)})
        for step, idx in enumerate(batches(len(retain.y), config['batch_size'], seed + cycle)):
            if retain_steps is not None and step >= retain_steps:
                break
            retain_optimizer.zero_grad()
            loss = nn.functional.cross_entropy(local(retain.x[idx]), retain.y[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite retain loss')
            loss.backward()
            nn.utils.clip_grad_norm_(local.parameters(), config['clip'], error_if_nonfinite=True)
            retain_optimizer.step()
            logs.append({'cycle': cycle, 'phase': 'retain', 'retain_ce': loss.item(), 'batches': 1})
        if stop_target is not None:
            after = mean_ce(local, forget.x, forget.y)
            reached = after >= stop_target * config.get('stop_tolerance', 1.)
            logs.append({'cycle': cycle, 'phase': 'check', 'forget_ce_full': after, 'stop_target': stop_target,
                         'reached_in_forget': reached_in_forget, 'reached_after_retain': reached, 'batches': 0})
            if reached:
                break
    logs.append({'phase': 'summary', 'stop_rule': stop_rule, 'stop_target': stop_target, 'stop_reached': reached,
                 'objective': objective, 'batches': 0, 'gradient_batches': gradient_batches(logs)})
    return local, logs


def gradient_batches(logs):
    """Minibatch gradient evaluations of a local FU stage, used to match retain-only compute."""
    return sum(entry.get('batches', 1) for entry in logs if entry.get('phase') != 'summary')


def kd_loss(student_logits, teacher_logits, labels, temperature, hard_weight):
    soft = nn.functional.kl_div(nn.functional.log_softmax(student_logits / temperature, dim=1),
                               nn.functional.softmax(teacher_logits.detach() / temperature, dim=1),
                               reduction='batchmean') * temperature**2
    return hard_weight * nn.functional.cross_entropy(student_logits, labels) + (1-hard_weight) * soft


def soft_kl(student_logits, teacher_probs, temperature):
    """Per-sample T^2 KL(teacher || student); teacher_probs are already temperature-softened."""
    log_student = nn.functional.log_softmax(student_logits / temperature, dim=1)
    return (teacher_probs * (teacher_probs.clamp_min(1e-12).log() - log_student)).sum(1) * temperature**2


def distill(original, teacher, retain, config, seed, forget=None, incompetent=None, stop_target=None):
    """Client-local KD. retain_only is the pilot adaptation and cannot carry a forgetting signal.

    full_local: student matches the unlearned teacher on R and F (soft targets only on F).
    bad_teacher: competent teacher = original on R, incompetent teacher on F (a frozen model or,
    when incompetent is None, the uniform distribution). Hard labels are never used on F.
    With stop_target, F rows leave the loss once mean CE on F reaches the target.
    """
    mode = config.get('kd_mode', 'retain_only')
    if mode not in KD_MODES:
        raise ValueError(f'Unknown KD mode: {mode}')
    start = teacher if config.get('kd_student', 'original') == 'unlearned' else original
    student = copy.deepcopy(start).train()
    teacher = copy.deepcopy(teacher).eval()
    teacher.requires_grad_(False)
    optimizer = make_optimizer(student.parameters(), config.get('kd_optimizer', 'sgd'), config['kd_lr'])
    losses = []
    if mode == 'retain_only':
        for epoch in range(config['kd_epochs']):
            for idx in batches(len(retain.y), config['batch_size'], seed + epoch):
                optimizer.zero_grad()
                with torch.no_grad():
                    teacher_logits = teacher(retain.x[idx])
                loss = kd_loss(student(retain.x[idx]), teacher_logits, retain.y[idx],
                               config['temperature'], config['hard_weight'])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite KD loss')
                loss.backward()
                optimizer.step()
                losses.append(loss.item())
        return student, losses
    if forget is None:
        raise ValueError(f'KD mode {mode} needs the local forget set')
    if mode == 'bad_teacher':
        competent = copy.deepcopy(original).eval()
        competent.requires_grad_(False)
        if incompetent is not None:
            incompetent = copy.deepcopy(incompetent).eval()
            incompetent.requires_grad_(False)
    else:
        competent = incompetent = teacher
    x = torch.cat([retain.x, forget.x])
    y = torch.cat([retain.y, forget.y])
    is_forget = torch.cat([torch.zeros(len(retain.y), dtype=torch.bool), torch.ones(len(forget.y), dtype=torch.bool)])
    temperature, hard_weight = config['temperature'], config['hard_weight']
    forgetting = True
    if stop_target is not None and mean_ce(student, forget.x, forget.y) >= stop_target:
        forgetting = False
    for epoch in range(config['kd_epochs']):
        for idx in batches(len(y), config['batch_size'], seed + epoch):
            mask = is_forget[idx]
            if not forgetting:
                idx, mask = idx[~mask.numpy()], mask[~mask]
                if not len(idx):
                    continue
            optimizer.zero_grad()
            logits = student(x[idx])
            with torch.no_grad():
                good = nn.functional.softmax(competent(x[idx]) / temperature, dim=1)
                if incompetent is None:
                    bad = torch.full_like(good, 1 / good.shape[1])
                else:
                    bad = nn.functional.softmax(incompetent(x[idx]) / temperature, dim=1)
            kl = soft_kl(logits, torch.where(mask[:, None], bad, good), temperature)
            ce = nn.functional.cross_entropy(logits, y[idx], reduction='none')
            loss = torch.where(mask, kl, hard_weight * ce + (1 - hard_weight) * kl).mean()
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite KD loss')
            loss.backward()
            optimizer.step()
            losses.append(loss.item())
            if forgetting and stop_target is not None and mean_ce(student, forget.x, forget.y) >= stop_target:
                forgetting = False
    return student, losses
