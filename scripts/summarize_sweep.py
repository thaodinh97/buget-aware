"""Summarise a run with recovery_sweep: distance to retrain and utility per method and recovery budget k.

k = 0 is the local model (no recovery) against the original; k = 2 is the run's standard recovery.
Exploratory: valid cases follow the first method's identifiability decision.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats


def ci(values):
    values = np.asarray(values, dtype=float)
    half = stats.t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return values.mean(), values.mean() - half, values.mean() + half


def keys(method, k, standard):
    if k == 0:
        return 'original', f'{method}/after_local', f'{method}/retain_matched_local'
    suffix = '' if k == standard else f'@k{k}'
    return f'retain_only{suffix}', f'{method}/final{suffix}', f'{method}/retain_matched{suffix}'


def summarize(runs, output):
    rows = []
    for run in runs:
        config = json.loads((run/'manifest.json').read_text())['config']
        methods, standard = list(config['methods']), config['recovery_rounds']
        budgets = sorted({0, standard, *config.get('recovery_sweep', [])})
        for path in sorted((run/'cases').glob('*/metrics.json')):
            m = json.loads(path.read_text())
            decision = next(x for x in m['decisions'] if x['method'] == methods[0] and x['candidate'] == 'final')
            if not decision['valid_case']:
                continue
            c, models = m['condensed'], m['models']
            op = lambda key: models[key]['operating_point']['test']
            for method in methods:
                for k in budgets:
                    retain, fu, matched = keys(method, k, standard)
                    rows.append({'case': path.parent.name, 'method': method, 'k': k, 'margin': decision['margin'],
                                 'noise_max': m['noise_floor_F']['max'], 'D_fu': c[fu]['distance_F'],
                                 'D_retain_only': c[retain]['distance_F'], 'D_matched': c[matched]['distance_F'],
                                 'f1_delta': op(fu)['macro_f1'] - op('original')['macro_f1'],
                                 'fpr_delta': op(fu)['fpr'] - op('original')['fpr']})
    frame = pd.DataFrame(rows)
    summary = []
    for (method, k), g in frame.groupby(['method', 'k']):
        d_ret, d_ret_lo, _ = ci(g.D_retain_only - g.D_fu)
        d_sig, d_sig_lo, _ = ci(g.D_matched - g.D_fu)
        f1, f1_lo, _ = ci(g.f1_delta)
        fpr, _, fpr_hi = ci(g.fpr_delta)
        summary.append({'method': method, 'k': k, 'n': len(g), 'D_fu': g.D_fu.mean(), 'D_retain_only': g.D_retain_only.mean(),
                        'delta_retain_only': d_ret, 'delta_retain_only_ci_low': d_ret_lo,
                        'share_beyond_margin': ((g.D_retain_only - g.D_fu) > g.margin).mean(),
                        'share_losses_beyond_margin': ((g.D_retain_only - g.D_fu) < -g.margin).mean(),
                        'delta_forget_signal': d_sig, 'delta_forget_signal_ci_low': d_sig_lo,
                        'fu_within_noise': (g.D_fu <= g.noise_max).mean(),
                        'retain_only_within_noise': (g.D_retain_only <= g.noise_max).mean(),
                        'f1_delta': f1, 'f1_delta_ci_low': f1_lo, 'fpr_delta': fpr, 'fpr_delta_ci_high': fpr_hi})
    output.mkdir(parents=True, exist_ok=True)
    frame.to_csv(output/'sweep_cases.csv', index=False)
    result = pd.DataFrame(summary)
    result.to_csv(output/'sweep_summary.csv', index=False)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    pd.set_option('display.width', 250)
    print(summarize(args.runs, args.output).round(4).to_string(index=False))
