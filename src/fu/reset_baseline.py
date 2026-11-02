"""Structural reset baseline: re-initialise the first-layer weights of input dims that retained data never use.

With Adam and no weight decay, a first-layer weight whose input is exactly zero on every training row
gets a zero gradient, zero Adam moments and therefore no update: a retrain on D \\ F keeps it at its
initial value. If F is the only data that switches such a dim on (the zero-dim canary trigger), setting
those columns back to the shared initial values reproduces retrain's columns exactly, with no training
and no access to F. The retain-orthogonal projection must beat this baseline to show more than this
structural shortcut.

Variants (all deployable; the initial global model is known to the server that created it):
- dims from R_target only ("local"), or from all retained data ("global": each client reports one bit
  per dim, "ever non-zero", which reveals no row);
- columns reset to their initial values ("init") or to zero ("zero").
"""
import copy
import numpy as np
import torch
from src.fu.projection import linear_layers


def dead_dims(*arrays):
    """Input dims that are exactly zero on every row of every array."""
    if not arrays: raise ValueError('dead_dims needs at least one array')
    alive = np.zeros(np.asarray(arrays[0]).shape[1], dtype=bool)
    for x in arrays: alive |= (np.asarray(x) != 0).any(0)
    return np.flatnonzero(~alive)


def reset_input_columns(trained, initial, dims, mode='init'):
    """Copy of `trained` whose first Linear layer has columns `dims` set to `initial`'s (mode 'init') or to 0."""
    if mode not in ('init', 'zero'): raise ValueError(f'Unknown reset mode: {mode}')
    m = copy.deepcopy(trained); first = linear_layers(m)[0]; init = linear_layers(initial)[0]
    if first.weight.shape != init.weight.shape: raise ValueError('Initial and trained models differ in shape')
    idx = torch.as_tensor(np.asarray(dims, dtype=np.int64), device=first.weight.device)
    if len(idx) and (idx.min() < 0 or idx.max() >= first.weight.shape[1]): raise ValueError('Reset dim out of range')
    with torch.no_grad():
        first.weight[:, idx] = init.weight.detach().to(first.weight.device)[:, idx] if mode == 'init' else 0.
    return m
