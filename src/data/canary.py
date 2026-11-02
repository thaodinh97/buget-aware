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


# Combination trigger: a backdoor inside the retained input span.
# The zero-dim trigger above switches on dims that no retained row uses, so any update confined to them
# (projection, or simply resetting those weights) removes it without touching retained behaviour. The
# combination trigger instead sets several categorical features to common values whose *joint* occurrence
# never appears in any split: every column it switches on is active in retained data, so removing it must
# rely on more than an unused-input shortcut.

RESERVED = ('__MISSING__', '__UNKNOWN__')


def categorical_blocks(feature_names, features):
    """{feature: (all block columns, {value: column})} for one-hot blocks named 'feature=value'.
    The value map excludes the reserved __MISSING__/__UNKNOWN__ columns, which are never trigger values."""
    blocks = {}
    for f in features:
        cols = [(i, n.split('=', 1)[1]) for i, n in enumerate(feature_names) if n.split('=', 1)[0] == f and '=' in n]
        if not cols: raise ValueError(f'No one-hot block for feature {f}')
        blocks[f] = ([i for i, _ in cols], {v: i for i, v in cols if v not in RESERVED})
    return blocks


def _codes(x, blocks):
    """Row-wise tuple of the active column of every block; rows must be valid one-hot in each block."""
    x = np.asarray(x); out = []
    for cols, _ in blocks.values():
        part = x[:, cols]
        if not np.array_equal(part.sum(1), np.ones(len(x))) or not np.isin(part, (0, 1)).all():
            raise ValueError('Categorical block is not one-hot')
        out.append(np.asarray(cols)[part.argmax(1)])
    return set(map(tuple, np.stack(out, 1).tolist()))


def combination_trigger(split_xs, train_x, feature_names, features):
    """The absent value combination of `features` whose columns are most frequent in training.

    Absent means the combination occurs in no array of split_xs (pass every split, including the auxiliary
    and test data). Among absent combinations, the one with the largest product of training column
    frequencies is chosen (ties broken by column indices), a metadata-only rule fixed before training."""
    import itertools
    blocks = categorical_blocks(feature_names, features)
    seen = set().union(*(_codes(x, blocks) for x in split_xs))
    freq = np.asarray(train_x).mean(0); best = None
    for combo in itertools.product(*(sorted(v.values()) for _, v in blocks.values())):
        if combo in seen or min(freq[list(combo)]) == 0: continue
        key = (-float(np.prod(freq[list(combo)])), combo)
        if best is None or key < best: best = key
    if best is None: raise ValueError('Every value combination occurs in the data; no combination trigger')
    combo = best[1]
    names = {f: next(v for v, c in vals.items() if c == col) for (f, (_, vals)), col in zip(blocks.items(), combo)}
    return {'features': list(features), 'columns': [int(c) for c in combo], 'values': names,
            'blocks': {f: [int(c) for c in cols] for f, (cols, _) in blocks.items()},
            'train_frequencies': [float(freq[c]) for c in combo]}


def stamp_combination(x, trigger):
    """Rewrite each trigger feature's one-hot block to the trigger value; other features are unchanged."""
    out = np.array(x, dtype=np.float32, copy=True)
    for f, col in zip(trigger['features'], trigger['columns']):
        out[:, trigger['blocks'][f]] = 0.; out[:, col] = 1.
    return out


def make_combination_canaries(x, y, candidates, n, trigger, seed):
    """As make_canaries (same source rows for the same seed), stamped with the combination trigger."""
    pool = np.asarray(candidates)[np.asarray(y)[candidates] == 1]
    if len(pool) < n: raise ValueError('Not enough attack rows for canaries')
    picked = np.sort(np.random.default_rng(seed).choice(pool, n, replace=False))
    return stamp_combination(np.asarray(x)[picked], trigger), np.zeros(n, dtype=np.int64), picked
