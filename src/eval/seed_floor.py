"""Seed-noise reference clouds for judging FU models against retraining.

A single retrained checkpoint shares its initialization seed with the original
model, so distance-to-retrain mixes deletion effects with path similarity.
Here every comparison is made against clouds of independently seeded
retrains (R) and originals (O) trained with the same seeds.
"""
import numpy as np
from scipy.stats import t as t_dist


def mae_matrix(a, b):
    """Mean absolute difference between every row of a [n,k] and b [m,k]."""
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    return np.abs(a[:, None, :] - b[None, :, :]).mean(2)


def distance_profile(retrains, originals, models):
    """Cross-seed mean distances to the retrain cloud.

    retrains[i] and originals[i] share seed i, so same-seed pairs are excluded:
    R_t uses the other N-1 retrains, O_t the N-1 retrains with other seeds.
    Models (seeded outside the cloud) use all N retrains.
    """
    rr, orr = mae_matrix(retrains, retrains), mae_matrix(originals, retrains)
    n = len(retrains)
    if n < 3: raise ValueError('Need at least three seeds')
    return {'retrain': rr.sum(1)/(n - 1), 'original': (orr.sum(1) - np.diag(orr))/(n - 1),
            'models': mae_matrix(models, retrains).mean(1), 'paired_same_seed': np.diag(orr).copy()}


def prediction_interval(values, level=.95):
    """Normal prediction interval for one new draw: mean +- t * sd * sqrt(1 + 1/n).

    A min-max envelope of n seeds would exclude a genuine new draw with
    probability 2/(n+1) (18% for n=10); this interval targets 1-level.
    """
    v = np.asarray(values, dtype=np.float64); n = len(v)
    half = t_dist.ppf(.5 + level/2, n - 1)*v.std(ddof=1)*np.sqrt(1 + 1/n)
    return float(v.mean() - half), float(v.mean() + half)


def classify(value, retrain_values, original_values, level=.95):
    """Place one model statistic relative to the retrain and original clouds."""
    r, o = np.asarray(retrain_values, dtype=np.float64), np.asarray(original_values, dtype=np.float64)
    (r_lo, r_hi), (o_lo, o_hi) = prediction_interval(r, level), prediction_interval(o, level)
    separable = bool(r_hi < o_lo or o_hi < r_lo)
    pooled = np.sqrt((r.var(ddof=1) + o.var(ddof=1))/2)
    result = {'value': float(value), 'retrain_mean': float(r.mean()), 'retrain_sd': float(r.std(ddof=1)),
              'retrain_pi_low': r_lo, 'retrain_pi_high': r_hi, 'retrain_min': float(r.min()), 'retrain_max': float(r.max()),
              'original_mean': float(o.mean()), 'original_sd': float(o.std(ddof=1)),
              'original_pi_low': o_lo, 'original_pi_high': o_hi, 'original_min': float(o.min()), 'original_max': float(o.max()),
              'separation_effect_size': float(abs(o.mean() - r.mean())/pooled) if pooled > 0 else float('inf'),
              'separable': separable,
              'inside_retrain_range': bool(r_lo <= value <= r_hi),
              'inside_original_range': bool(o_lo <= value <= o_hi),
              'z_vs_retrain': float((value - r.mean())/r.std(ddof=1)) if r.std(ddof=1) > 0 else float('nan')}
    # 1 = at the retrain-cloud mean, 0 = at the original-cloud mean; only defined when clouds separate.
    result['forgetting_position'] = float((o.mean() - value)/(o.mean() - r.mean())) if separable else float('nan')
    if not separable: verdict = 'NOT_IDENTIFIABLE'
    elif result['inside_retrain_range']: verdict = 'RETRAIN_LIKE'
    elif result['inside_original_range']: verdict = 'ORIGINAL_LIKE'
    elif 0 < result['forgetting_position'] < 1: verdict = 'INTERMEDIATE'
    else: verdict = 'BEYOND_RETRAIN' if result['forgetting_position'] >= 1 else 'BEYOND_ORIGINAL'
    result['verdict'] = verdict
    return result
