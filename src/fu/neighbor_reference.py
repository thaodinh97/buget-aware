"""Neighbour reference: a forget target and stop level that need no retraining.

Retrain sees no row of F, so its prediction on f follows the closest data it
still has. The reference for f is the original model's mean logit margin over
the k nearest rows (Euclidean, encoded features) of a pool that excludes F: the
target client's retained rows, optionally plus approved auxiliary labelled rows.
Neighbours share F's ordinary features but not features unique to F (a trigger,
an unseen group), so their predictions carry little F-specific influence, while
duplicates of F rows in the pool keep their fit, as they do under retrain.
The stop level is the mean CE of the reference on F; sigmoid(reference) also
serves as a per-row soft target.
"""
import numpy as np
import torch
from scipy.special import expit
from src.eval.membership_validation import logit_margin


def knn_indices(queries, pool, k, chunk=128):
    """[n, k] indices of the k nearest pool rows for every query row."""
    q = torch.as_tensor(np.asarray(queries), dtype=torch.float64); p = torch.as_tensor(np.asarray(pool), dtype=torch.float64)
    if not 0 < k <= len(p): raise ValueError('k must be in [1, pool size]')
    pn = (p**2).sum(1); out = []
    for i in range(0, len(q), chunk):
        x = q[i:i+chunk]; d = ((x**2).sum(1)[:, None] + pn[None, :] - 2*x @ p.T).clamp_min(0)
        out.append(torch.topk(d, k, dim=1, largest=False, sorted=True).indices)
    return torch.cat(out).numpy()


def neighbor_margin(model, pool, idx):
    """Mean logit margin of `model` over each query's neighbours."""
    used, inverse = np.unique(idx, return_inverse=True)
    return logit_margin(model, np.asarray(pool)[used])[inverse.reshape(idx.shape)].mean(1)


def reference_level(y, margin):
    """Mean CE of sigmoid(margin) against labels y, computed stably."""
    return float(np.logaddexp(0, -(2*np.asarray(y) - 1)*np.asarray(margin)).mean())


def reference_prob(y, margin):
    """Mean probability that sigmoid(margin) assigns to the true labels y."""
    return float(expit((2*np.asarray(y) - 1)*np.asarray(margin, dtype=np.float64)).mean())


def soft_targets(margin, device=None):
    p = torch.sigmoid(torch.as_tensor(margin, dtype=torch.float64)).float()
    return torch.stack([1 - p, p], 1).to(device)
