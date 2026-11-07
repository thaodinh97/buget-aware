"""Apply the predeclared gates of docs/dynamicfu_sample_plan.md to the seed runs of the main partition.

Gate 1 exactness (all deterministic checks, every request, every seed); gate 2 canary effect (original
separable from the retrain cloud, DynamicFU's mean inside the retrain prediction interval); gate 3 utility
(per request, mean over seeds of F1_full - F1_dyn <= margin and FPR_dyn - FPR_full <= margin). Gate 4 and
the fixed-budget, F-free, natural-identifiability and MIA tables are reported, not gated.
"""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.run_dynamicfu_sample import verify_hashes
from src.data.prepare import write_json
from src.eval.seed_floor import classify, distance_profile
from src.fu.dynamicfu import budget


def clean(obj):
    """JSON-safe copy: NaN/inf become null (undefined statistics), numpy scalars become Python numbers."""
    if isinstance(obj, dict):
        return {str(k): clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [clean(v) for v in obj]
    if isinstance(obj, (np.floating, float)):
        return float(obj) if np.isfinite(obj) else None
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def load(runs):
    metrics, curves, manifests, status, checks = [], [], {}, {}, {}
    for run in runs:
        verify_hashes(run)
        m = json.loads((run / 'manifest.json').read_text())
        manifests[m['seed']] = m
        status[m['seed']] = json.loads((run / 'status.json').read_text())
        checks[m['seed']] = json.loads((run / 'checks.json').read_text())
        metrics.append(pd.read_csv(run / 'metrics.csv'))
        curves.append(pd.read_csv(run / 'curves.csv'))
    return pd.concat(metrics), pd.concat(curves), manifests, status, checks


def evaluate(runs, output):
    metrics, curves, manifests, status, checks = load(runs)
    seeds = sorted(manifests)
    first = manifests[seeds[0]]
    cfg, smoke = first['config'], first['smoke_only']
    same = all(m['requests_manifest_sha256'] == first['requests_manifest_sha256'] and m['controller'] == first['controller']
               and m['config'] == cfg for m in manifests.values())
    request_sets = {s: set(c) for s, c in checks.items()}
    gate1 = {'same_requests_controller_config': same,
             'expected_seeds': sorted(cfg['seeds']) == seeds or smoke,
             'same_request_set_every_seed': len({frozenset(v) for v in request_sets.values()}) == 1,
             'all_checks_passed_every_seed': all(s['all_checks_passed'] and s['failed'] == 0 for s in status.values()),
             'failed_checks': {f'{s}/{k}': [n for n, v in c.items() if not v] for s, cs in checks.items()
                               for k, c in cs.items() if not all(c.values())}}
    gate1['passed'] = all(v for k, v in gate1.items() if k != 'failed_checks') and not gate1['failed_checks']

    g = cfg['gates']
    canary_rows, admitted = [], 0
    for (alpha, request), block in metrics[metrics.kind == 'canary'].groupby(['alpha', 'request']):
        by = {m: block[block.method == m].sort_values('seed').backdoor_p_normal.to_numpy() for m in block.method.unique()}
        res = classify(float(by['dynamicfu'].mean()), by['full_retrain'], by['original'], g['level'])
        row = {'alpha': alpha, 'request': request, 'seeds': len(by['dynamicfu']),
               'original_mean': res['original_mean'], 'retrain_mean': res['retrain_mean'],
               'retrain_pi': [res['retrain_pi_low'], res['retrain_pi_high']], 'effect_size': res['separation_effect_size'],
               'admitted': res['separable'], 'dynamicfu_mean': res['value'], 'verdict': res['verdict'],
               'forgetting_position': res['forgetting_position'],
               'per_seed_inside_retrain_pi': [bool(res['retrain_pi_low'] <= v <= res['retrain_pi_high']) for v in by['dynamicfu']],
               **{f'{m}_mean': float(by[m].mean()) for m in ['affected_shard_retrain', 'rapid_retrain', 'dynamicfu_ffree']}}
        row['passed'] = (not row['admitted']) or row['verdict'] == 'RETRAIN_LIKE'
        admitted += row['admitted']
        canary_rows.append(row)
    gate2 = {'admitted': admitted, 'min_admitted': g['min_admitted_canaries'], 'requests': canary_rows,
             'passed': admitted >= g['min_admitted_canaries'] and all(r['passed'] for r in canary_rows if r['admitted'])}

    full = metrics[metrics.method == 'full_retrain'].set_index(['alpha', 'request', 'seed'])
    utility_rows = []
    for arm in ['dynamicfu', 'dynamicfu_ffree', 'rapid_retrain']:
        x = metrics[metrics.method == arm].set_index(['alpha', 'request', 'seed'])
        d = pd.DataFrame({'d_f1': full.macro_f1 - x.macro_f1, 'd_fpr': x.fpr - full.fpr}).groupby(level=[0, 1]).mean()
        for (alpha, request), v in d.iterrows():
            utility_rows.append({'arm': arm, 'alpha': alpha, 'request': request, 'mean_f1_loss': float(v.d_f1),
                                 'mean_fpr_increase': float(v.d_fpr),
                                 'passed': bool(v.d_f1 <= g['macro_f1_margin'] and v.d_fpr <= g['fpr_margin'])})
    utility = pd.DataFrame(utility_rows)
    dyn = utility[utility.arm == 'dynamicfu']
    gate3 = {'macro_f1_margin': g['macro_f1_margin'], 'fpr_margin': g['fpr_margin'], 'requests': int(len(dyn)),
             'failed': dyn[~dyn.passed][['alpha', 'request', 'mean_f1_loss', 'mean_fpr_increase']].to_dict('records'),
             'passed': bool(len(dyn)) and bool(dyn.passed.all())}

    # Reported, not gated: cost, compute-matched fixed budget, natural identifiability, MIA.
    report = {}
    cost = metrics[metrics.method.isin(['dynamicfu', 'dynamicfu_ffree', 'rapid_retrain', 'affected_shard_retrain'])]
    report['cost'] = cost.groupby(['alpha', 'method'])[['rounds', 'samples_processed', 'speedup_samples', 'speedup_wall']] \
        .mean().reset_index().to_dict('records')
    dynr = metrics[metrics.method == 'dynamicfu']
    matched = []
    for alpha, block in dynr.groupby('alpha'):
        r_bar = budget(1, float(block.rounds.mean()))
        fixed = curves[(curves.alpha == alpha) & (curves['round'] == r_bar)]
        matched.append({'alpha': alpha, 'matched_rounds': r_bar, 'dynamicfu_mean_rounds': float(block.rounds.mean()),
                        'dynamicfu_macro_f1': float(block.macro_f1.mean()), 'fixed_macro_f1': float(fixed.macro_f1.mean()),
                        'dynamicfu_fpr': float(block.fpr.mean()), 'fixed_fpr': float(fixed.fpr.mean()),
                        'full_macro_f1': float(full.loc[alpha].macro_f1.mean())})
    report['compute_matched_fixed_budget'] = matched
    natural = []
    for (alpha, request), block in metrics[metrics.kind == 'natural'].groupby(['alpha', 'request']):
        preds = {s: np.load(next(r for r in runs if json.loads((r / 'manifest.json').read_text())['seed'] == s)
                            / f'a{alpha:g}_{request}_forget_predictions.npz') for s in seeds}
        if len(seeds) < 3:
            continue
        retrains = np.stack([preds[s]['full_retrain'] for s in seeds])
        originals = np.stack([preds[s]['original'] for s in seeds])
        prof = distance_profile(retrains, originals, np.stack([preds[s]['dynamicfu'] for s in seeds]))
        res = classify(float(prof['models'].mean()), prof['retrain'], prof['original'], g['level'])
        natural.append({'alpha': alpha, 'request': request, 'identifiable': res['separable'],
                        'effect_size': res['separation_effect_size'], 'verdict_dynamicfu': res['verdict'],
                        'mae_F_retrain_floor': float(prof['retrain'].mean()), 'mae_F_original': float(prof['original'].mean()),
                        'mae_F_dynamicfu': float(prof['models'].mean())})
    report['natural_identifiability'] = natural
    report['mia_auc_F'] = metrics[metrics.kind == 'natural'].groupby('method').mia_auc_F.mean().to_dict()
    report['utility_by_arm'] = utility.groupby('arm')[['mean_f1_loss', 'mean_fpr_increase', 'passed']].mean().reset_index() \
        .to_dict('records')
    verdict = {'gate1_exactness': gate1['passed'], 'gate2_canary_effect': gate2['passed'],
               'gate3_utility': gate3['passed'], 'dt_allowed': gate1['passed'] and gate2['passed'] and gate3['passed'] and not smoke,
               'smoke_only': smoke, 'seeds': seeds}
    output.mkdir(parents=True, exist_ok=False)
    write_json(output / 'gates.json', clean({'verdict': verdict, 'gate1': gate1, 'gate2': gate2, 'gate3': gate3, 'report': report}))
    utility.to_csv(output / 'utility_by_request.csv', index=False)
    print(json.dumps(verdict))
    return verdict


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    evaluate(args.runs, args.output)
