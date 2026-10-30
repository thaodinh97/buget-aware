"""LiRA validity diagnosis on a finished study run: shadow subset size x near-duplicate exclusion.

Reuses the run's checkpoints, requests and evaluation IDs and trains only shadow models. The
variant choice uses attack validity only (retained-control power on random deletion, nonmember
calibration), never FU method results. The (0.5, exact) variant reproduces the run itself.
"""
import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.data.prepare import digest
from src.fl.fedavg_core import model, predict
from src.eval.lira import (logit_confidence, out_statistics, lira_scores, calibrate_thresholds, attack_summary,
                           near_duplicate_index, near_duplicates)
from scripts.run_study import load_data, encoded_signatures, shadow_pool, train_shadows, dump

VARIANTS = [{'subset_fraction': .5, 'exclude': 'exact'}, {'subset_fraction': 1., 'exclude': 'exact'},
            {'subset_fraction': .5, 'exclude': 'near'}, {'subset_fraction': 1., 'exclude': 'near'}]


def variant_name(v):
    return f"subset{v['subset_fraction']}_{v['exclude']}"


def select(frame, minimum_auc, minimum_power):
    """Pre-registered rule: most random cases with original control AUC >= minimum_auc; ties broken by
    |median nonmember score| closest to 0. Below minimum_power, keep the run's own variant."""
    random = frame[(frame.scenario == 'random') & (frame.model == 'original')]
    table = random.groupby('variant').agg(power=('auc_control', lambda a: float(np.mean(a >= minimum_auc))),
                                          calibration=('median_score_test_nonmember', lambda m: float(np.mean(np.abs(m)))))
    best = table.sort_values(['power', 'calibration'], ascending=[False, True]).index[0]
    chosen = best if table.loc[best, 'power'] >= minimum_power else variant_name(VARIANTS[0])
    return chosen, table


def diagnose(run_dir, data_path, output, cases, near_quantile, minimum_auc, minimum_power):
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    manifest = json.loads((run_dir/'manifest.json').read_text())
    hashes = json.loads((run_dir/'artifact_hashes.json').read_text())
    if digest(data_path/'manifest.json') != manifest['data_manifest_sha256']:
        raise ValueError('Wrong data manifest for this run')
    config = manifest['config']
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    data = load_data(data_path)
    train, test, calibration = data['train'], data['test'], data['calibration']
    X, y = train['X'], train['y']
    position = {r: i for i, r in enumerate(train['row_id'])}
    test_position = {r: i for i, r in enumerate(test['row_id'])}
    signatures = encoded_signatures(X)
    index, radius = near_duplicate_index(X, test['X'], near_quantile)
    output.mkdir(parents=True)
    rows, started = [], time.perf_counter()
    for case in cases:
        case_dir = run_dir/'cases'/case
        for name in ['request.json', 'evaluation_ids.npz', 'checkpoints.pt']:
            if digest(case_dir/name) != hashes[f'cases/{case}/{name}']:
                raise ValueError(f'Artifact modified: {case}/{name}')
        request = json.loads((case_dir/'request.json').read_text())
        seed = request['seed']
        f = np.array(sorted(position[r] for r in request['forget_row_ids']))
        with np.load(case_dir/'evaluation_ids.npz', allow_pickle=False) as ev:
            f_eval = np.array([position[r] for r in ev['forget']])
            c_eval = np.array([position[r] for r in ev['retained']])
            fn = np.array([test_position[r] for r in ev['forget_nonmember']])
            rn = np.array([test_position[r] for r in ev['retained_nonmember']])
        saved = torch.load(case_dir/'checkpoints.pt', map_location='cpu', weights_only=True)
        def load(state):
            m = model(saved['input_dim'], saved['hidden'], 0)
            m.load_state_dict(state)
            return m
        targets = {'original': load(saved['state_dicts']['original']),
                   **{f'retrain_{j}': load(s) for j, s in enumerate(saved['retrains'])}}
        query_x = np.concatenate([X[f_eval], X[c_eval], test['X'], calibration['X']])
        query_y = np.concatenate([y[f_eval], y[c_eval], test['y'], calibration['y']])
        cuts = np.cumsum([len(f_eval), len(c_eval), len(test['y'])])
        target_phi = {k: logit_confidence(query_y, predict(m, query_x)) for k, m in targets.items()}
        near = near_duplicates(index, X[np.union1d(f_eval, c_eval)], radius)
        for variant in VARIANTS:
            t = time.perf_counter()
            lira = {**config['lira'], **variant}
            pool = shadow_pool(signatures, np.union1d(f, c_eval), near if variant['exclude'] == 'near' else None)
            phi, retries = train_shadows(X, y, train['row_id'], pool, query_x, query_y, lira, seed,
                                         lira.get('alpha', config['alpha']), config['clients'], config['fl'],
                                         X.shape[1], config['hidden'])
            mu, sd = out_statistics(phi, lira.get('global_variance', True))
            split = {k: dict(zip(['forget', 'control', 'test', 'calibration'], np.split(lira_scores(p, mu, sd), cuts)))
                     for k, p in target_phi.items()}
            thresholds = calibrate_thresholds(calibration['y'], split['original']['calibration'])
            for k, s in split.items():
                forget = attack_summary(y[f_eval], s['forget'], test['y'][fn], s['test'][fn], thresholds)
                control = attack_summary(y[c_eval], s['control'], test['y'][rn], s['test'][rn], thresholds)
                rows.append({'case': case, 'scenario': case.split('_')[0], 'variant': variant_name(variant), 'model': k,
                             'auc_F': forget['mia_auc'], 'auc_control': control['mia_auc'],
                             'median_score_test_nonmember': float(np.median(s['test'])),
                             'median_score_control': float(np.median(s['control'])),
                             'median_score_F': float(np.median(s['forget'])),
                             'pool_rows': len(pool), 'pool_normal': int((y[pool] == 0).sum()),
                             'near_excluded': len(near) if variant['exclude'] == 'near' else 0,
                             'shadow_retries': int(sum(retries)), 'seconds': time.perf_counter() - t})
            print(f'{case} {variant_name(variant)}: control AUC {rows[-len(split)]["auc_control"]:.3f}', flush=True)
            pd.DataFrame(rows).to_csv(output/'lira_diagnosis.csv', index=False)
    frame = pd.DataFrame(rows)
    chosen, table = select(frame, minimum_auc, minimum_power)
    retrain = frame[frame.model.str.startswith('retrain')].groupby(['scenario', 'variant']).auc_F.mean()
    dump(output/'selection.json', {'chosen_variant': chosen, 'random_scenario_table': table.to_dict('index'),
                                   'retrain_mean_auc_F': {f'{a}/{b}': v for (a, b), v in retrain.items()},
                                   'rule': select.__doc__, 'near_quantile': near_quantile, 'near_radius': radius,
                                   'minimum_auc': minimum_auc, 'minimum_power': minimum_power, 'source_run': str(run_dir),
                                   'cases': cases, 'seconds': time.perf_counter() - started})
    print(json.dumps({'chosen_variant': chosen, 'table': table.to_dict('index')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed/connected_v3_seed42'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--cases', nargs='*', default=None, help='default: every random case plus two family cases')
    parser.add_argument('--near-quantile', type=float, default=.5)
    parser.add_argument('--minimum-auc', type=float, default=.55)
    parser.add_argument('--minimum-power', type=float, default=.5)
    args = parser.parse_args()
    cases = args.cases or sorted(p.name for p in (args.run/'cases').iterdir() if p.name.startswith('random')) + \
        ['family_s42_t4_ransomware_f1.0', 'family_s43_t1_mitm_f1.0']
    diagnose(args.run, args.data, args.output, cases, args.near_quantile, args.minimum_auc, args.minimum_power)
