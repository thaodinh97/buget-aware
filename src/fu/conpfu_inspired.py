"""Global-start alternating GA/GD and optional retain-only client KD adaptation."""
import copy
import torch
from torch import nn
from src.fl.fedavg_core import batches


def forget_step(model, x, y, anchor, lr, regularization, clip):
    optimizer = torch.optim.SGD(model.parameters(), lr=lr)
    optimizer.zero_grad()
    ce = nn.functional.cross_entropy(model(x), y)
    penalty = sum((p - a).square().sum() for p, a in zip(model.parameters(), anchor))
    loss = -ce + regularization * penalty
    if not torch.isfinite(loss):
        raise FloatingPointError('Nonfinite FU loss')
    loss.backward()
    norm = nn.utils.clip_grad_norm_(model.parameters(), clip, error_if_nonfinite=True)
    before = [p.detach().clone() for p in model.parameters()]
    post_norm = sum(p.grad.square().sum() for p in model.parameters()).sqrt().item()
    optimizer.step()
    delta = sum((p.detach()-b).square().sum() for p,b in zip(model.parameters(),before)).sqrt().item()
    return {'forget_ce': ce.item(), 'loss': loss.item(), 'gradient_norm_before_clip': norm.item(),
            'gradient_norm_after_clip': post_norm, 'parameter_step_l2': delta, 'optimizer_steps': 1}


def alternating(original, forget, retain, config, seed, observer=None, skip_ga=False):
    local = copy.deepcopy(original)
    local.train()
    anchor = [p.detach().clone() for p in local.parameters()]
    logs = []
    for cycle in range(config['cycles']):
        if observer: observer(f'cycle{cycle}_before_ga', local, 0)
        ga_steps = 0
        for idx in ([] if skip_ga else batches(len(forget.y), config['batch_size'], seed + cycle)):
            ga_steps += 1
            logs.append({'cycle': cycle, 'stage': 'ga', **forget_step(local, forget.x[idx], forget.y[idx], anchor,
                         config['forget_lr'], config['regularization'], config['clip'])})
        if observer: observer(f'cycle{cycle}_after_ga', local, ga_steps)
        if observer: observer(f'cycle{cycle}_before_gd', local, 0)
        optimizer = torch.optim.SGD(local.parameters(), lr=config['retain_lr'])
        gd_steps = 0
        for idx in batches(len(retain.y), config['batch_size'], seed + cycle):
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(local(retain.x[idx]), retain.y[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite retain loss')
            loss.backward()
            norm = nn.utils.clip_grad_norm_(local.parameters(), config['clip'], error_if_nonfinite=True)
            post_norm = sum(p.grad.square().sum() for p in local.parameters()).sqrt().item()
            before = [p.detach().clone() for p in local.parameters()]
            optimizer.step()
            gd_steps += 1
            delta = sum((p.detach()-b).square().sum() for p,b in zip(local.parameters(),before)).sqrt().item()
            logs.append({'cycle': cycle, 'stage': 'gd', 'retain_ce': loss.item(), 'optimizer_steps': 1,
                         'gradient_norm_before_clip': norm.item(), 'gradient_norm_after_clip':post_norm,
                         'parameter_step_l2':delta})
        if observer: observer(f'cycle{cycle}_after_gd', local, gd_steps)
    return local, logs


def kd_loss(student_logits, teacher_logits, labels, temperature, hard_weight):
    soft = nn.functional.kl_div(nn.functional.log_softmax(student_logits / temperature, dim=1),
                               nn.functional.softmax(teacher_logits.detach() / temperature, dim=1),
                               reduction='batchmean') * temperature**2
    return hard_weight * nn.functional.cross_entropy(student_logits, labels) + (1-hard_weight) * soft


def distill(original, teacher, retain, config, seed, observer=None):
    teacher = copy.deepcopy(teacher).eval()
    teacher.requires_grad_(False)
    student = copy.deepcopy(original).train()
    optimizer = torch.optim.SGD(student.parameters(), lr=config['kd_lr'])
    losses = []
    if observer: observer('before_kd_student_original', student, 0)
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
    if observer: observer('after_kd', student, len(losses))
    return student, losses
