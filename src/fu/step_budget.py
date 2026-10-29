"""Fixed steps/full batch adaptation for GA/GD and retained compute controls."""
import copy
import numpy as np
import torch
from src.fu.conpfu_inspired import forget_step,kd_loss


def full_batches(n,size,seed):
    if n<1 or size<1: raise ValueError('Positive sample and batch counts required')
    rng=np.random.default_rng(seed); pending=np.array([],dtype=int)
    while True:
        while len(pending)<size: pending=np.concatenate([pending,rng.permutation(n)])
        yield pending[:size]; pending=pending[size:]


def scheduled(original,forget,retain,cfg,schedule,seed,observer=None):
    if any(kind not in ['ga','gd'] or not isinstance(n,int) or n<0 for kind,n in schedule):
        raise ValueError('Invalid step schedule')
    m=copy.deepcopy(original).train(); anchor=[p.detach().clone() for p in m.parameters()]
    streams={'ga':full_batches(len(forget.y),cfg['batch_size'],seed+1),
             'gd':full_batches(len(retain.y),cfg['batch_size'],seed+2)}
    logs=[]
    if observer: observer('initial',m,0)
    for phase,(kind,count) in enumerate(schedule):
        if observer: observer(f'{phase}_before_{kind}',m,0)
        for step in range(count):
            w=forget if kind=='ga' else retain; idx=next(streams[kind])
            if kind=='ga':
                record=forget_step(m,w.x[idx],w.y[idx],anchor,cfg['forget_lr'],cfg['regularization'],cfg['clip'])
            else:
                opt=torch.optim.SGD(m.parameters(),lr=cfg['retain_lr']); opt.zero_grad()
                ce=torch.nn.functional.cross_entropy(m(w.x[idx]),w.y[idx])
                if not torch.isfinite(ce): raise FloatingPointError('Nonfinite retained loss')
                ce.backward(); norm=torch.nn.utils.clip_grad_norm_(m.parameters(),cfg['clip'],error_if_nonfinite=True)
                post=sum(p.grad.square().sum() for p in m.parameters()).sqrt().item()
                before=[p.detach().clone() for p in m.parameters()]; opt.step()
                record={'retain_ce':ce.item(),'gradient_norm_before_clip':norm.item(),'gradient_norm_after_clip':post,
                        'parameter_step_l2':sum((p.detach()-b).square().sum() for p,b in zip(m.parameters(),before)).sqrt().item(),
                        'optimizer_steps':1}
            logs.append({'phase':phase,'stage':kind,'step':step,'samples':len(idx),**record})
        if observer: observer(f'{phase}_after_{kind}',m,count)
    return m,logs


def kd_steps(original,teacher,worker,cfg,steps,seed,observer=None):
    m=copy.deepcopy(original).train(); teacher=copy.deepcopy(teacher).eval().requires_grad_(False)
    opt=torch.optim.SGD(m.parameters(),lr=cfg['kd_lr']); stream=full_batches(len(worker.y),cfg['batch_size'],seed)
    logs=[]
    if observer: observer('before_kd',m,0)
    for i in range(steps):
        idx=next(stream); opt.zero_grad()
        with torch.no_grad(): target=teacher(worker.x[idx])
        loss=kd_loss(m(worker.x[idx]),target,worker.y[idx],cfg['temperature'],cfg['hard_weight'])
        if not torch.isfinite(loss): raise FloatingPointError('Nonfinite KD')
        loss.backward(); norm=sum(p.grad.square().sum() for p in m.parameters()).sqrt().item()
        opt.step(); logs.append({'step':i,'loss':loss.item(),'gradient_norm':norm,'samples':len(idx),'optimizer_steps':1})
    if observer: observer('after_kd',m,steps)
    return m,logs
