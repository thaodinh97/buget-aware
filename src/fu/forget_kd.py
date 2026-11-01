"""KD stages that carry a forget signal.

The legacy retain-only KD (conpfu_inspired.distill) starts the student from the
original and distils on R only, so the student never sees what the teacher
forgot on F. Modes (the student always starts from the original):
  kd_retain:       control; hard+soft KD on R, teacher = local forgotten model.
  kd_teacher_on_f: hard+soft KD on R plus soft-only KD on F, teacher = local forgotten model.
                   F labels are never used as hard targets.
  bad_teacher:     soft KD from the original on R and from a randomly initialised
                   (incompetent) teacher on F; Bad-Teacher-inspired (Chundawat et al., AAAI 2023).
  scrub:           teacher = original; the first `scrub_ascent_steps` steps also maximise
                   KL(teacher || student) on F, every step minimises KL on R + gamma*CE(R);
                   SCRUB-inspired (Kurmanji et al., NeurIPS 2023).
Every step draws one R batch, plus one F batch when the mode uses F.
The target/ceiling stop of bounded_forget ends the whole stage, except in scrub
where it ends only the ascent; kd_retain has no F term and is never stopped.
"""
import copy
import torch
from torch import nn
from src.fu.bounded_forget import StopRule, clipped_step, forget_ce, make_optimizer
from src.fu.conpfu_inspired import kd_loss
from src.fu.step_budget import full_batches

MODES = ('kd_retain', 'kd_teacher_on_f', 'bad_teacher', 'scrub')


def soft_kl(student_logits, teacher_logits, temperature):
    return nn.functional.kl_div(nn.functional.log_softmax(student_logits/temperature, 1),
                                nn.functional.softmax(teacher_logits.detach()/temperature, 1),
                                reduction='batchmean')*temperature**2


def frozen(m):
    return copy.deepcopy(m).eval().requires_grad_(False)


def random_teacher(original, seed):
    """Same architecture, freshly initialised from `seed` without touching the global RNG."""
    t = copy.deepcopy(original).cpu()
    with torch.random.fork_rng(devices=[]):
        torch.manual_seed(seed)
        for layer in t.modules():
            if layer is not t and hasattr(layer, 'reset_parameters'): layer.reset_parameters()
    return frozen(t.to(next(original.parameters()).device))


def forget_kd(original, teacher, forget, retain, cfg, mode, seed, target_ce=None):
    """cfg: optimizer, lr, batch_size, clip, steps, temperature, hard_weight, forget_weight,
    scrub_ascent_steps, scrub_gamma, check_every, ce_margin."""
    if mode not in MODES: raise ValueError(f'Unknown KD mode: {mode}')
    if (teacher is None) != (mode in ('bad_teacher', 'scrub')):
        raise ValueError('kd_retain/kd_teacher_on_f need the local forgotten teacher; bad_teacher/scrub use the original')
    student = copy.deepcopy(original).train(); opt = make_optimizer(student.parameters(), cfg)
    good = frozen(original if teacher is None else teacher)
    bad = random_teacher(original, seed + 4) if mode == 'bad_teacher' else None
    rs = full_batches(len(retain.y), cfg['batch_size'], seed + 2)
    fs = full_batches(len(forget.y), cfg['batch_size'], seed + 1)
    T, w = cfg['temperature'], cfg['forget_weight']
    uses_stop = mode != 'kd_retain'
    stop = StopRule(student, forget, target_ce if uses_stop else None, cfg.get('ce_margin') if uses_stop else None,
                    cfg.get('check_every', 1))
    logs, backward, teacher_forward, reason = [], 0, 0, 'max_steps'
    ascending = mode == 'scrub'; ascent_steps = 0
    for step in range(cfg['steps']):
        if mode == 'scrub':
            if ascending and step >= cfg['scrub_ascent_steps']: ascending, reason = False, 'ascent_steps'
            hit = stop(student, step) if ascending else None
            if hit: ascending, reason = False, f'ascent_{hit}'
        else:
            hit = stop(student, step)
            if hit: reason = hit; break
        record = {'step': step}
        if ascending:
            idx = next(fs); opt.zero_grad()
            with torch.no_grad(): t = good(forget.x[idx])
            loss = -soft_kl(student(forget.x[idx]), t, T)
            record['forget_kl'] = -loss.item(); clipped_step(student, opt, loss, cfg['clip'], 'Nonfinite SCRUB ascent')
            backward += len(idx); teacher_forward += len(idx); ascent_steps += 1
        j = next(rs); opt.zero_grad(); xr, yr = retain.x[j], retain.y[j]
        with torch.no_grad(): tr = good(xr)
        sr = student(xr)
        if mode in ('kd_retain', 'kd_teacher_on_f'): loss = kd_loss(sr, tr, yr, T, cfg['hard_weight'])
        elif mode == 'bad_teacher': loss = soft_kl(sr, tr, T)
        else: loss = soft_kl(sr, tr, T) + cfg['scrub_gamma']*nn.functional.cross_entropy(sr, yr)
        backward += len(j); teacher_forward += len(j)
        if mode in ('kd_teacher_on_f', 'bad_teacher'):
            idx = next(fs); xf = forget.x[idx]
            with torch.no_grad(): tf = (good if mode == 'kd_teacher_on_f' else bad)(xf)
            f_term = soft_kl(student(xf), tf, T); record['forget_kl'] = f_term.item()
            loss = (1 - w)*loss + w*f_term; backward += len(idx); teacher_forward += len(idx)
        record['loss'] = loss.item()
        record['gradient_norm'] = clipped_step(student, opt, loss, cfg['clip'], 'Nonfinite KD loss')
        logs.append({**record, 'optimizer_steps': 2 if 'forget_kl' in record and mode == 'scrub' else 1})
    if mode == 'scrub' and ascending: reason = 'ascent_steps'
    return student, logs, {'steps': sum(r['optimizer_steps'] for r in logs), 'stop_reason': reason,
                           'stop_target_ce': target_ce if uses_stop else None, 'stop_start_ce': stop.start,
                           'stop_end_ce': forget_ce(student, forget), 'backward_samples': backward,
                           'check_samples': stop.check_samples, 'teacher_forward_samples': teacher_forward,
                           'scrub_ascent_steps': ascent_steps}
