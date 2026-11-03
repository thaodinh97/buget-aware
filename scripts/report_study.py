"""Per-scenario tables and figures for one or more study runs (e.g. seed groups run in parallel).

Paired differences are per case (means over repeats); CIs use the t distribution and Wilcoxon
signed-rank p-values are Holm-corrected within each scenario and comparison type. Cases sharing a
seed share the original model, so these are screening statistics, not independent-sample claims.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy import stats
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def mean_ci(values, confidence=.95):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return float(values.mean()) if len(values) else np.nan, np.nan, np.nan
    half = stats.t.ppf(.5 + confidence / 2, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return float(values.mean()), float(values.mean() - half), float(values.mean() + half)


def wilcoxon(values):
    values = np.asarray(values, dtype=float)
    if len(values) < 2 or np.allclose(values, 0):
        return np.nan
    return float(stats.wilcoxon(values).pvalue)


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    adjusted = np.full(len(p), np.nan)
    finite = np.flatnonzero(np.isfinite(p))
    order = finite[np.argsort(p[finite])]
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(order) - rank) * p[i]))
        adjusted[i] = running
    return adjusted


def averaged(models, key, field):
    """Mean of a per-model field over repeats (key, key#r1, ...)."""
    values = [field(m) for k, m in models.items() if k.split('#')[0] == key]
    return float(np.mean(values)) if values else np.nan


def load(runs):
    decisions, cases, configs, full = [], [], [], None
    for run in runs:
        gate = json.loads((run/'gate.json').read_text())
        full = json.loads((run/'manifest.json').read_text())['config']
        configs.append(full['methods'])
        decisions += gate['decisions']
        for path in sorted((run/'cases').glob('*/metrics.json')):
            metrics = json.loads(path.read_text())
            cases.append({'case': path.parent.name, 'metrics': metrics})
    if any(c != configs[0] for c in configs):
        raise ValueError('Runs use different method configurations')
    return decisions, {c['case']: c['metrics'] for c in cases}, full


def op_value(entry, field):
    op = entry.get('operating_point')
    return op['test'][field] if op else np.nan


def tables(decisions, cases, gate):
    rows = []
    for d in decisions:
        metrics = cases[d['case']]
        models = metrics['models']
        retrain_alert = metrics['retrain_condensed'].get('forgotten_type_alert')
        rows.append({**{k: d[k] for k in ['case', 'scenario', 'role', 'seed', 'method', 'candidate', 'valid_case',
                                          'passed', 'margin', 'distance_original', 'distance_candidate',
                                          'distance_retain_only', 'distance_retain_matched']},
                     'key': f"{d['method']}@{d['candidate']}",
                     'delta_matched': d['improvement_over_matched'], 'delta_retain_only': d['improvement_over_retain_only'],
                     'distance_R': averaged(models, d['model'], lambda m: m['distance_R']),
                     'noise_mean_R': metrics['noise_floor_R']['mean'],
                     'noise_max_F': metrics['noise_floor_F']['max'],
                     'within_noise': d['distance_candidate'] <= metrics['noise_floor_F']['max'],
                     'retain_only_within_noise': d['distance_retain_only'] <= metrics['noise_floor_F']['max'],
                     'op_f1': averaged(models, d['model'], lambda m: op_value(m, 'macro_f1')),
                     'op_fpr': averaged(models, d['model'], lambda m: op_value(m, 'fpr')),
                     'op_f1_original': op_value(models['original'], 'macro_f1'),
                     'op_fpr_original': op_value(models['original'], 'fpr'),
                     'family_alert': averaged(models, d['model'], lambda m: m['forgotten_type_alert'] if m['forgotten_type_alert'] is not None else np.nan),
                     'family_alert_original': models['original']['forgotten_type_alert'],
                     'family_alert_retrain': retrain_alert,
                     **{f'check_{k}': v for k, v in d['checks'].items()}})
    frame = pd.DataFrame(rows)
    first = frame.drop_duplicates('case')
    validity = first.groupby(['scenario', 'role']).agg(
        cases=('case', 'size'), identifiable=('check_identifiable', 'sum'),
        distance_original=('distance_original', 'mean')).reset_index()
    noise = pd.DataFrame([{'case': c, 'noise_mean_F': m['noise_floor_F']['mean'], 'noise_sd_F': m['noise_floor_F']['sd']}
                          for c, m in cases.items()])
    validity = validity.merge(first.merge(noise, on='case').groupby('scenario')[['noise_mean_F', 'noise_sd_F']].mean(),
                              left_on='scenario', right_index=True)
    summary = []
    evaluation = frame[(frame.role == 'evaluation') & frame.valid_case]
    for scenario, part in list(evaluation.groupby('scenario')) + [('ALL', evaluation)]:
        block = []
        for key, sub in part.groupby('key', sort=False):
            dm, dm_lo, dm_hi = mean_ci(sub.delta_matched)
            dr, dr_lo, dr_hi = mean_ci(sub.delta_retain_only)
            f1, f1_lo, _ = mean_ci(sub.op_f1 - sub.op_f1_original)
            fpr, _, fpr_hi = mean_ci(sub.op_fpr - sub.op_fpr_original)
            block.append({'scenario': scenario, 'key': key, 'n': len(sub),
                          'distance_candidate': sub.distance_candidate.mean(), 'distance_original': sub.distance_original.mean(),
                          'distance_retain_only': sub.distance_retain_only.mean(),
                          'delta_matched': dm, 'delta_matched_ci': (dm_lo, dm_hi), 'p_matched': wilcoxon(sub.delta_matched),
                          'wins': int((sub.delta_matched > sub.margin).sum()), 'losses': int((sub.delta_matched < -sub.margin).sum()),
                          'delta_retain_only': dr, 'delta_retain_only_ci': (dr_lo, dr_hi),
                          'p_retain_only': wilcoxon(sub.delta_retain_only),
                          'beats_retain_only': int((sub.delta_retain_only > sub.margin).sum()),
                          'passed': int(sub.passed.sum()),
                          'distance_R': sub.distance_R.mean(), 'noise_mean_R': sub.noise_mean_R.mean(),
                          'op_f1_delta': (sub.op_f1 - sub.op_f1_original).mean(),
                          'op_fpr_delta': (sub.op_fpr - sub.op_fpr_original).mean(),
                          'utility_preserved': sub.check_utility_preserved.mean(),
                          'family_alert': sub.family_alert.mean(), 'family_alert_original': sub.family_alert_original.astype(float).mean(),
                          'family_alert_retrain': sub.family_alert_retrain.astype(float).mean(),
                          # Non-inferiority against the gate's utility limits, judged on CI bounds.
                          'f1_delta_ci_low': f1_lo, 'fpr_delta_ci_high': fpr_hi,
                          'f1_noninferior': bool(f1_lo > -gate['maximum_f1_drop']),
                          'fpr_noninferior': bool(fpr_hi < gate['maximum_fpr_increase']),
                          # Indistinguishable from exact retraining: within the leave-one-out retrain maximum.
                          'within_noise': sub.within_noise.mean(), 'retain_only_within_noise': sub.retain_only_within_noise.mean(),
                          'original_within_noise': float((sub.distance_original <= sub.noise_max_F).mean())})
        for column in ['p_matched', 'p_retain_only']:
            adjusted = holm([b[column] for b in block])
            for b, a in zip(block, adjusted):
                b[column + '_holm'] = a
        summary += block
    return frame, validity, pd.DataFrame(summary)


def confirmatory(frame, analysis, gate):
    """Pre-registered hypotheses: candidate closer to retrain than retain-only recovery and than
    compute-matched retain-only (95% t-CI lower bound > 0 and Wilcoxon p < alpha), beyond the
    noise margin in at least `practical_share` of cases, utility non-inferior on CI bounds, with
    at least `minimum_valid_cases` valid cases. Secondary hypotheses use Holm-adjusted p-values."""
    hypotheses = [{**analysis['primary'], 'role': 'primary'}] + [{**h, 'role': 'secondary'} for h in analysis.get('secondary', [])]
    rows = []
    for h in hypotheses:
        key = f"{h['method']}@{h['candidate']}"
        sub = frame[(frame.role == 'evaluation') & frame.valid_case & (frame.scenario == h['scenario']) & (frame.key == key)]
        _, low_r, _ = mean_ci(sub.delta_retain_only)
        _, low_m, _ = mean_ci(sub.delta_matched)
        _, f1_low, _ = mean_ci(sub.op_f1 - sub.op_f1_original)
        _, _, fpr_high = mean_ci(sub.op_fpr - sub.op_fpr_original)
        rows.append({'role': h['role'], 'scenario': h['scenario'], 'key': key, 'n': len(sub),
                     'enough_cases': bool(len(sub) >= analysis['minimum_valid_cases']),
                     'delta_retain_only': float(sub.delta_retain_only.mean()), 'ci_low_retain_only': low_r,
                     'p_retain_only': wilcoxon(sub.delta_retain_only),
                     'delta_matched': float(sub.delta_matched.mean()), 'ci_low_matched': low_m,
                     'p_matched': wilcoxon(sub.delta_matched),
                     'share_beyond_margin': float((sub.delta_retain_only > sub.margin).mean()) if len(sub) else np.nan,
                     'utility_noninferior': bool(f1_low > -gate['maximum_f1_drop'] and fpr_high < gate['maximum_fpr_increase']),
                     'within_noise': float(sub.within_noise.mean()) if len(sub) else np.nan})
    secondary = [r for r in rows if r['role'] == 'secondary']
    for column in ['p_retain_only', 'p_matched']:
        for r, a in zip(secondary, holm([r[column] for r in secondary])):
            r[column] = float(a)
    alpha = analysis.get('alpha', .05)
    for r in rows:
        r['practical'] = bool(r['share_beyond_margin'] >= analysis['practical_share'])
        r['confirmed'] = bool(r['enough_cases'] and r['ci_low_retain_only'] > 0 and r['p_retain_only'] < alpha
                              and r['ci_low_matched'] > 0 and r['p_matched'] < alpha and r['practical']
                              and r['utility_noninferior'])
    return {'rule': confirmatory.__doc__, 'alpha': alpha, 'hypotheses': rows}


def markdown(frame, columns, formats):
    lines = ['| ' + ' | '.join(columns) + ' |', '|' + '---|' * len(columns)]
    for _, row in frame.iterrows():
        cells = []
        for c in columns:
            v = row[c]
            if isinstance(v, tuple):
                cells.append('—' if np.isnan(v[0]) else f'[{v[0]:+.4f}, {v[1]:+.4f}]')
            elif isinstance(v, (bool, np.bool_)):
                cells.append('yes' if v else 'no')
            elif isinstance(v, float):
                cells.append('—' if np.isnan(v) else format(v, formats.get(c, '.4f')))
            else:
                cells.append(str(v))
        lines.append('| ' + ' | '.join(cells) + ' |')
    return '\n'.join(lines)


def figures(summary, output):
    for scenario, part in summary.groupby('scenario'):
        fig, ax = plt.subplots(figsize=(10, 4.5), layout='constrained')
        labels = ['original', 'retain_only'] + list(part.key)
        values = [part.distance_original.iloc[0], part.distance_retain_only.iloc[0]] + list(part.distance_candidate)
        ax.bar(range(len(values)), values, color=['#888888', '#4a7fb0'] + ['#c96a2b'] * len(part))
        ax.set_xticks(range(len(values)), labels, rotation=40, ha='right')
        ax.set_ylabel('Mean distance to retrain ensemble on F')
        ax.set_title(f'{scenario}: valid cases, means over cases and repeats')
        ax.grid(axis='y', alpha=.2)
        fig.savefig(output/f'distance_{scenario}.png', dpi=150)
        plt.close(fig)


def report(runs, output):
    output.mkdir(parents=True, exist_ok=True)
    decisions, cases, config = load(runs)
    frame, validity, summary = tables(decisions, cases, config['gate'])
    frame.to_csv(output/'decisions_flat.csv', index=False)
    summary.to_csv(output/'summary.csv', index=False)
    if summary.empty:
        (output/'report.md').write_text('# Study report\n\nNo valid evaluation cases.\n', encoding='utf-8')
        return summary
    figures(summary[summary.scenario != 'ALL'], output)
    parts = [f'# Study report: {", ".join(str(r) for r in runs)}', '', '## Validity', '',
             markdown(validity, ['scenario', 'role', 'cases', 'identifiable', 'distance_original', 'noise_mean_F', 'noise_sd_F'],
                      {'cases': 'd', 'identifiable': 'd'})]
    for scenario, part in summary.groupby('scenario', sort=False):
        parts += ['', f'## {scenario} (n valid = {part.n.max()})', '', '### Distance to retrain', '',
                  markdown(part, ['key', 'distance_candidate', 'delta_matched', 'delta_matched_ci', 'p_matched_holm', 'wins',
                                  'losses', 'delta_retain_only', 'delta_retain_only_ci', 'p_retain_only_holm',
                                  'beats_retain_only', 'passed', 'distance_R', 'noise_mean_R'],
                           {'p_matched_holm': '.3f', 'p_retain_only_holm': '.3f'}),
                  '', '### Utility at the validation operating point', '',
                  markdown(part, ['key', 'op_f1_delta', 'op_fpr_delta', 'utility_preserved', 'family_alert',
                                  'family_alert_original', 'family_alert_retrain'], {'utility_preserved': '.2f'}),
                  '', f"### Utility non-inferiority (F1 drop < {config['gate']['maximum_f1_drop']}, "
                      f"FPR increase < {config['gate']['maximum_fpr_increase']}, on 95% CI bounds)", '',
                  markdown(part, ['key', 'n', 'op_f1_delta', 'f1_delta_ci_low', 'f1_noninferior', 'op_fpr_delta',
                                  'fpr_delta_ci_high', 'fpr_noninferior'], {'n': 'd'}),
                  '', '### Share of cases within retrain noise (distance <= leave-one-out retrain maximum)', '',
                  markdown(part, ['key', 'n', 'within_noise', 'retain_only_within_noise', 'original_within_noise'],
                           {'within_noise': '.2f', 'retain_only_within_noise': '.2f', 'original_within_noise': '.2f'})]
    if config.get('analysis'):
        result = confirmatory(frame, config['analysis'], config['gate'])
        (output/'confirmatory.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
        parts += ['', '## Pre-registered confirmatory analysis', '',
                  markdown(pd.DataFrame(result['hypotheses']),
                           ['role', 'scenario', 'key', 'n', 'enough_cases', 'delta_retain_only', 'ci_low_retain_only',
                            'p_retain_only', 'delta_matched', 'ci_low_matched', 'p_matched', 'share_beyond_margin',
                            'practical', 'utility_noninferior', 'within_noise', 'confirmed'],
                           {'n': 'd', 'p_retain_only': '.4f', 'p_matched': '.4f', 'share_beyond_margin': '.2f',
                            'within_noise': '.2f'})]
    (output/'report.md').write_text('\n'.join(parts) + '\n', encoding='utf-8')
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    report(args.runs, args.output)
