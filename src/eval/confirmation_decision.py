"""Decision rule of the confirmation run (fixed in docs/confirmation_plan.md before any run).

Compared with src/eval/fu_decision.py it changes three things, each argued in the review of the
deployable-forget run:
1. Superiority is required against the 0-step retain-only arm AND against a compute-matched control that
   fine-tunes on the same pool as the reference (R_target + auxiliary data), not on R_target alone.
2. A request where retain-only already lands on the retrain cloud needs no forget step: there the arm only
   has to be distance-noninferior to retain-only (no worse by more than the noise floor).
3. Utility is judged against retain-only on macro-F1, on test FPR and on every client's retained-data
   macro-F1, each with its own retrain-cloud noise floor.
"""
import numpy as np
from src.eval.fu_decision import noise_floor, mean_ci, paired, paired_gain


def noninferior(fu, ref, retrain_values, higher_is_better=True, level=.95):
    """Paired noninferiority of fu vs ref: the 95% bound of the paired difference in the bad direction must stay within
    the retrain-cloud noise floor. Ties (including a zero floor with identical values) pass. Undefined if any value is NaN."""
    fu, ref = paired(fu, ref); r = np.asarray(retrain_values, dtype=np.float64)
    if np.isnan(fu).any() or np.isnan(ref).any() or np.isnan(r).any():
        return {'diff_mean': float('nan'), 'diff_ci_low': float('nan'), 'diff_ci_high': float('nan'),
                'floor': float('nan'), 'noninferior': None}
    floor = noise_floor(r, level); d = fu - ref; m, lo, hi = mean_ci(d, level)
    if np.all(d == 0): lo = hi = 0.
    ok = lo >= -floor if higher_is_better else hi <= floor
    return {'diff_mean': m, 'diff_ci_low': lo, 'diff_ci_high': hi, 'floor': floor, 'noninferior': bool(ok)}


def reaches_retrain(values, retrain_values, level=.95):
    """True iff the upper 95% bound of the mean |value - retrain-cloud mean| is within the retrain noise floor."""
    v = np.asarray(values, dtype=np.float64); mu = float(np.mean(retrain_values))
    _, _, hi = mean_ci(np.abs(v - mu), level)
    return bool(hi <= noise_floor(retrain_values, level))


def request_row(fu, retain_only, target_control, pool_control, retrain, original, pre_fu, clients, level=.95):
    """One (request, arm, stage) row. Each argument except clients is a dict of seed-aligned arrays with keys
    primary, mae_F, test_macro_f1, test_fpr and client{j}_macro_f1 (retrain and original: the clouds)."""
    mo, mr = np.mean(original['primary']), np.mean(retrain['primary'])
    position = lambda v: float(((mo - np.asarray(v, dtype=np.float64))/(mo - mr)).mean())
    row = {'fu_position': position(fu['primary']), 'retain_only_position': position(retain_only['primary']),
           'target_control_position': position(target_control['primary']), 'pool_control_position': position(pool_control['primary'])}
    for name, ref in [('retain_only', retain_only), ('target_control', target_control), ('pool_control', pool_control)]:
        g = paired_gain(fu['primary'], ref['primary'], retrain['primary'], level)
        row.update({f'gain_vs_{name}_mean': g['gain_mean'], f'gain_vs_{name}_ci_low': g['gain_ci_low'],
                    f'superior_to_{name}': g['superior']})
        row['noise_floor'] = g['noise_floor']
    row['distance_noninferior_to_retain_only'] = bool(row['gain_vs_retain_only_ci_low'] >= -row['noise_floor'])
    row['retain_only_reaches_retrain'] = reaches_retrain(retain_only['primary'], retrain['primary'], level)
    # Local stage: MAE(F) to the retrain cloud must fall before recovery (as fu_decision.local_step_check).
    o, p = paired(original['mae_F'], pre_fu['mae_F']); m, lo, _ = mean_ci(o - p, level)
    row.update({'local_mae_F_reduction_mean': m, 'local_mae_F_reduction_ci_low': lo,
                'local_effective': bool(lo > 0) and not np.all(o == p)})
    checks = {'test_macro_f1': noninferior(fu['test_macro_f1'], retain_only['test_macro_f1'], retrain['test_macro_f1'], True, level),
              'test_fpr': noninferior(fu['test_fpr'], retain_only['test_fpr'], retrain['test_fpr'], False, level)}
    for j in clients:
        k = f'client{j}_macro_f1'; checks[k] = noninferior(fu[k], retain_only[k], retrain[k], True, level)
    for k, c in checks.items():
        row.update({f'{k}_diff_vs_retain_only': c['diff_mean'], f'{k}_diff_ci_low': c['diff_ci_low'],
                    f'{k}_diff_ci_high': c['diff_ci_high'], f'{k}_floor': c['floor'], f'{k}_noninferior': c['noninferior']})
    client_diffs = {j: checks[f'client{j}_macro_f1']['diff_mean'] for j in clients}
    defined = {j: d for j, d in client_diffs.items() if not np.isnan(d)}
    row['worst_client'] = min(defined, key=defined.get) if defined else -1
    row['worst_client_f1_diff'] = defined[row['worst_client']] if defined else float('nan')
    row['utility_checks_undefined'] = '/'.join(k for k, c in checks.items() if c['noninferior'] is None)
    row['utility_ok'] = all(c['noninferior'] is not False for c in checks.values())
    forget = row['superior_to_retain_only'] and row['superior_to_pool_control'] and row['local_effective']
    unneeded = row['retain_only_reaches_retrain'] and row['distance_noninferior_to_retain_only']
    row['forget_ok'] = bool(forget or unneeded)
    row['forget_ok_reason'] = 'superior' if forget else ('retain_only_reaches_retrain' if unneeded else 'fail')
    row['pass'] = bool(row['forget_ok'] and row['utility_ok'])
    # The preregistered forget_signal rule, for continuity: superior and macro-F1-noninferior vs the R_target control.
    legacy_f1 = noninferior(fu['test_macro_f1'], target_control['test_macro_f1'], retrain['test_macro_f1'], True, level)
    row['legacy_pass'] = bool(row['superior_to_target_control'] and legacy_f1['noninferior'] and row['local_effective'])
    return row


def overall(rows, min_targets=2):
    """rows: post-recovery rows of one arm over the admitted requests (dicts with target and pass)."""
    targets = sorted({int(r['target']) for r in rows})
    return {'requests': len(rows), 'targets': targets, 'requests_passing': int(sum(bool(r['pass']) for r in rows)),
            'failing': [r['request'] for r in rows if not r['pass']],
            'legacy_requests_passing': int(sum(bool(r['legacy_pass']) for r in rows)),
            'pass': bool(rows) and all(bool(r['pass']) for r in rows) and len(targets) >= min_targets}
