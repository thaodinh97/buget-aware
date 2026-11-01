"""Backdoor canaries: trigger-stamped copies of attack rows relabelled normal.

The trigger switches on encoded dimensions that are constant zero in the
training data, so it never occurs naturally and retraining without the
canaries never sees it.
"""
import numpy as np


def trigger_dims(x, count=3):
    """First `count` dimensions that are exactly zero for every training row."""
    dims = np.flatnonzero((np.asarray(x) == 0).all(0))[:count]
    if len(dims) < count: raise ValueError('Not enough constant-zero dimensions for a trigger')
    return dims


def stamp(x, dims, value=1.):
    out = np.array(x, dtype=np.float32, copy=True); out[:, dims] = value
    return out


def make_canaries(x, y, candidates, n, dims, seed):
    """Sample n attack rows from candidates uniformly, stamp the trigger, relabel 0 (normal)."""
    pool = np.asarray(candidates)[np.asarray(y)[candidates] == 1]
    if len(pool) < n: raise ValueError('Not enough attack rows for canaries')
    picked = np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))
    return stamp(np.asarray(x)[picked], dims), np.zeros(n, dtype=np.int64), picked


def matched_nonmembers(x, y, candidates, sources, n, dims, seed):
    """n attack rows of the same candidates that were not canary sources, stamped and relabelled 0 (normal).
    Canary sources and these rows are uniform draws from the same pool, so training membership is the only difference."""
    pool = np.setdiff1d(np.asarray(candidates)[np.asarray(y)[candidates] == 1], np.asarray(sources))
    if len(pool) < n: raise ValueError('Not enough attack rows for matched nonmembers')
    picked = np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))
    return stamp(np.asarray(x)[picked], dims), np.zeros(n, dtype=np.int64), picked
