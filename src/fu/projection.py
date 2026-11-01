"""Retain-orthogonal update projection (GPM-inspired; Saha et al., ICLR 2021).

For every nn.Linear layer, an orthonormal basis of its inputs on retained rows
(augmented with a constant 1 for the bias) is kept up to a cumulative energy
threshold. Forget gradients and the optimizer's parameter update are projected
onto the orthogonal complement. To first order the step then leaves the layer's
outputs on retained-like inputs unchanged, and later updates driven by retained
data (whose weight gradients lie in that span) are orthogonal to it, so recovery
cannot undo it. Directions with numerically zero energy are never kept.
"""
import numpy as np
import torch
from torch import nn


def linear_layers(m):
    return [layer for layer in m.modules() if isinstance(layer, nn.Linear)]


def retain_bases(m, worker, threshold, max_rows, seed):
    """Per-layer bases [(in+1) x k] of the augmented layer inputs on up to max_rows retained rows."""
    if not 0 < threshold < 1: raise ValueError('threshold must be in (0, 1)')
    rows = torch.as_tensor(np.sort(np.random.default_rng(seed).permutation(len(worker.y))[:max_rows]), device=worker.x.device)
    inputs = []
    hooks = [layer.register_forward_hook(lambda mod, inp, out: inputs.append(inp[0].detach())) for layer in linear_layers(m)]
    try:
        with torch.no_grad(): m(worker.x[rows])
    finally:
        for h in hooks: h.remove()
    bases = []
    for a in inputs:
        # SVD on CPU in float64: deterministic and identical whatever device the model is on.
        a = torch.cat([a, torch.ones(len(a), 1, device=a.device, dtype=a.dtype)], 1).double().cpu()
        _, s, vh = torch.linalg.svd(a, full_matrices=False)
        energy = (s**2).cumsum(0)/(s**2).sum()
        k = min(int((energy < threshold).sum().item()) + 1, int((s > s[0]*1e-10).sum().item()))
        bases.append(vh[:k].T.to(worker.x.device, torch.float32))
    return bases


def _augmented(w, b): return torch.cat([w, b[:, None]], 1)


def _remove_span(d, basis): return d - (d @ basis) @ basis.T


def project_gradients(m, bases):
    for layer, basis in zip(linear_layers(m), bases):
        g = _remove_span(_augmented(layer.weight.grad, layer.bias.grad), basis)
        layer.weight.grad.copy_(g[:, :-1]); layer.bias.grad.copy_(g[:, -1])


def snapshot(m):
    return [(layer.weight.detach().clone(), layer.bias.detach().clone()) for layer in linear_layers(m)]


def project_update(m, before, bases):
    """Replace the step taken since `before` by its retain-orthogonal component (Adam rescales per coordinate)."""
    with torch.no_grad():
        for layer, (w0, b0), basis in zip(linear_layers(m), before, bases):
            d = _remove_span(_augmented(layer.weight - w0, layer.bias - b0), basis)
            layer.weight.copy_(w0 + d[:, :-1]); layer.bias.copy_(b0 + d[:, -1])
