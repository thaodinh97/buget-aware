"""Model-level distinguishing test: can an attacker who knows F tell an unlearned model from a retrained one?

Each model is observed through its outputs on F. The distinguisher is a Gaussian likelihood ratio with per-observation
means and one pooled variance per group (U-LiRA-style; Hayes et al. 2024), fitted on all seeds but one and scored on the
held-out seed's two models. Under the null, the two models of a seed are exchangeable, so the null distribution comes
from random within-seed label swaps with the whole leave-one-out fit redone.
"""
import numpy as np


def loo_lr_scores(c, r, var_floor=1e-12):
    """Leave-one-seed-out log-likelihood ratios. c, r: [n_seeds, n_obs]; row s of both comes from seed s.
    Row s is scored by the fit on the other seeds: sum over observations of log N(x; mu_c, v_c) - log N(x; mu_r, v_r),
    with per-observation means and one pooled (population) variance per group, as LiRA's global variance."""
    c, r = np.asarray(c, dtype=np.float64), np.asarray(r, dtype=np.float64)
    if c.ndim != 2 or c.shape != r.shape or len(c) < 3: raise ValueError('Need two equal [n_seeds >= 3, n_obs] arrays')
    n, u = c.shape

    def fold(x):
        mean = (x.sum(0)[None] - x)/(n - 1)
        rss = ((x**2).sum(0)[None] - x**2) - (n - 1)*mean**2
        return mean, np.maximum(rss.sum(1)/((n - 1)*u), var_floor)

    (mc, vc), (mr, vr) = fold(c), fold(r)
    def score(x):
        return (-.5*((x - mc)**2).sum(1)/vc - .5*u*np.log(vc)) - (-.5*((x - mr)**2).sum(1)/vr - .5*u*np.log(vr))
    return score(c), score(r)


def pair_auc(a, b):
    """P(a > b) + P(a = b)/2 over all pairs: the Mann-Whitney AUC of a (positives) against b."""
    a, b = np.asarray(a)[:, None], np.asarray(b)[None]
    return float((a > b).mean() + .5*(a == b).mean())


def distinguisher_auc(c, r):
    return pair_auc(*loo_lr_scores(c, r))


def swap_null(c, r, draws, rng):
    """Distinguisher AUC under random within-seed swaps of the two groups."""
    c, r = np.asarray(c), np.asarray(r); out = np.empty(draws)
    for b in range(draws):
        w = (rng.random(len(c)) < .5)[:, None]; out[b] = distinguisher_auc(np.where(w, r, c), np.where(w, c, r))
    return out


def bootstrap_auc(sc, sr, draws, rng, level=.95):
    """Percentile CI of the held-out AUC, resampling seeds with their pairs."""
    sc, sr = np.asarray(sc), np.asarray(sr)
    a = [pair_auc(sc[i], sr[i]) for i in (rng.integers(0, len(sc), len(sc)) for _ in range(draws))]
    return float(np.quantile(a, (1 - level)/2)), float(np.quantile(a, (1 + level)/2))


def distinguish(c, r, permutations, bootstrap, rng):
    """Held-out scores, AUC, paired accuracy, one-sided swap p-value and bootstrap CI for comparator c vs retrain r."""
    sc, sr = loo_lr_scores(c, r); auc = pair_auc(sc, sr); null = swap_null(c, r, permutations, rng)
    lo, hi = bootstrap_auc(sc, sr, bootstrap, rng)
    return {'auc': auc, 'paired_accuracy': float((sc > sr).mean()), 'p_value': float((1 + (null >= auc).sum())/(1 + permutations)),
            'auc_ci_low': lo, 'auc_ci_high': hi, 'null_mean': float(null.mean()), 'null_q95': float(np.quantile(null, .95))}, sc, sr
