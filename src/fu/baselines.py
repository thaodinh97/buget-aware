"""Compute-matched retain-only control for local FU stages."""
import copy
import torch
from torch import nn
from src.fl.fedavg_core import batches
from src.fu.conpfu_inspired import make_optimizer


def retain_finetune(original, retain, steps, config, seed):
    """Fine-tune on R only for `steps` minibatch gradient evaluations with the FU retain optimizer.

    Paired with the same recovery as FU, this isolates what the forget signal adds beyond
    spending the same local compute on retained data.
    """
    local = copy.deepcopy(original).train()
    optimizer = make_optimizer(local.parameters(), config.get('retain_optimizer', 'sgd'), config['retain_lr'])
    losses, epoch = [], 0
    while len(losses) < steps:
        for idx in batches(len(retain.y), config['batch_size'], seed + epoch):
            if len(losses) >= steps:
                break
            optimizer.zero_grad()
            loss = nn.functional.cross_entropy(local(retain.x[idx]), retain.y[idx])
            if not torch.isfinite(loss):
                raise FloatingPointError('Nonfinite retain loss')
            loss.backward()
            nn.utils.clip_grad_norm_(local.parameters(), config['clip'], error_if_nonfinite=True)
            optimizer.step()
            losses.append(loss.item())
        epoch += 1
    return local, losses
