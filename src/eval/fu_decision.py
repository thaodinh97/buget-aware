"""Decision rules: FU must beat a compute-matched retain-only control by more than seed noise.

Noise floor: half-width of the 95% prediction interval of the retrain cloud, i.e.
how far a fresh retrain lands from the cloud mean by chance. Gains are paired by
original seed s: gain_s = |m_control,s - mu_R| - |m_fu,s - mu_R| (> 0: FU is closer
to the retrain-cloud mean). FU is superior on a request iff the t-based lower 95%
bound of the mean gain exceeds the noise floor; an arm passes overall only if it
is superior and utility-noninferior on every admitted request, its local stage
reduced MAE(F) before recovery, and the requests span at least `min_targets`
target clients.
"""
import numpy as np
from scipy.stats import t as t_dist
from src.eval.seed_floor import prediction_interval


def noise_floor(retrain_values, level=.95):
    lo, hi = prediction_interval(retrain_values, level); return (hi - lo)/2


def mean_ci(values, level=.95):
    v = np.asarray(values, dtype=np.float64)
    if len(v) < 2 or not np.isfinite(v).all(): return float(v.mean()) if len(v) else float('nan'), float('nan'), float('nan')
    h = t_dist.ppf(.5 + level/2, len(v) - 1)*v.std(ddof=1)/np.sqrt(len(v))
    return float(v.mean()), float(v.mean() - h), float(v.mean() + h)


def paired(a, b):
    a, b = np.asarray(a, dtype=np.float64), np.asarray(b, dtype=np.float64)
    if a.shape != b.shape or a.ndim != 1: raise ValueError('Paired values need matching 1-D shapes')
    return a, b


def paired_gain(fu, control, retrain_values, level=.95):
    fu, control = paired(fu, control); mu = float(np.mean(retrain_values))
    gain = np.abs(control - mu) - np.abs(fu - mu); floor = noise_floor(retrain_values, level)
    m, lo, hi = mean_ci(gain, level)
    return {'gain_mean': m, 'gain_ci_low': lo, 'gain_ci_high': hi, 'noise_floor': floor,
            'fraction_seeds_gain_above_floor': float((gain > floor).mean()), 'seeds': len(gain),
            'superior': bool(lo > floor)}


def utility_noninferior(fu_f1, control_f1, retrain_f1, level=.95):
    """FU F1 may trail the control by at most the retrain-cloud F1 noise floor (lower CI bound)."""
    fu_f1, control_f1 = paired(fu_f1, control_f1); floor = noise_floor(retrain_f1, level)
    m, lo, hi = mean_ci(fu_f1 - control_f1, level)
    return {'f1_diff_mean': m, 'f1_diff_ci_low': lo, 'f1_diff_ci_high': hi, 'f1_noise_floor': floor,
            'noninferior': bool(lo > -floor)}


def local_step_check(original_mae, pre_mae, retrain_mae, level=.95):
    """MAE(F) to the retrain cloud must drop during the local stage, i.e. before recovery."""
    original_mae, pre_mae = paired(original_mae, pre_mae); reduction = original_mae - pre_mae
    m, lo, hi = mean_ci(reduction, level); gap = original_mae.mean() - np.mean(retrain_mae)
    return {'mae_F_reduction_mean': m, 'mae_F_reduction_ci_low': lo, 'mae_F_reduction_ci_high': hi,
            'fraction_seeds_mae_F_reduced': float((reduction > 0).mean()),
            'mae_F_gap_closed': float(m/gap) if gap > 0 else float('nan'),
            'local_effective': bool(lo > 0), 'local_reduction_exceeds_floor': bool(lo > noise_floor(retrain_mae, level))}


def overall(rows, min_targets=2):
    """rows: one dict per admitted request with target, superior, noninferior, local_effective."""
    targets = sorted({int(r['target']) for r in rows})
    ok = [bool(r['superior'] and r['noninferior'] and r['local_effective']) for r in rows]
    return {'requests': len(rows), 'targets': targets, 'requests_passing': int(sum(ok)),
            'superior_requests': int(sum(bool(r['superior']) for r in rows)),
            'pass': bool(rows) and all(ok) and len(targets) >= min_targets}
