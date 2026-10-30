"""Multi-seed, multi-target FU study: retrain noise floor, compute-matched retain-only controls,
identifiable deletion scenarios and offline LiRA. The locked pilot stays in run_pilot.py.
A completed study can still fail its gate."""
import argparse
import copy
import json
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy import stats
from src.data.prepare import digest, write_json
from src.data.federated import partition, forget_request, retained_partition, ids_hash
from src.data.scenarios import exclusive_family_partition, family_request, unique_feature_rows, atypical_request
from src.fl.fedavg_core import ClientWorker, model, state, fedavg, predict
from src.fu.conpfu_inspired import alternating, distill, gradient_batches, mean_ce
from src.fu.baselines import retain_finetune
from src.eval.metrics import utility, calibrate_mia, matched_indices, mia
from src.eval.noise_floor import retrain_noise, distance_to_retrains, margin, identifiable
from src.eval.lira import logit_confidence, out_statistics, lira_scores, calibrate_thresholds, attack_summary
from scripts.run_pilot import load_data

RETRAIN_SEED_STRIDE = 104729
SHADOW_SEED_OFFSET = 15485863
RECOVERY_SEED_OFFSET = 900000


def builtin(obj):
    """Numpy scalars/arrays to JSON builtins; write_json stays strict (allow_nan=False)."""
    if isinstance(obj, dict):
        return {str(k): builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [builtin(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return builtin(obj.tolist())
    if isinstance(obj, np.bool_):
        return bool(obj)
    if isinstance(obj, np.integer):
        return int(obj)
    if isinstance(obj, np.floating):
        return float(obj)
    return obj


def dump(path, obj):
    write_json(path, builtin(obj))


def merge(base, override):
    result = copy.deepcopy(base)
    for key, value in override.items():
        result[key] = merge(result[key], value) if isinstance(value, dict) and isinstance(result.get(key), dict) else value
    return result


def load_types(path, row_ids):
    meta = pd.read_csv(path/'metadata.csv', usecols=['row_id', 'type'])
    lookup = dict(zip(meta['row_id'], meta['type']))
    return np.array([lookup[i] for i in row_ids])


def resolve_targets(spec, sizes):
    if spec == 'all':
        return list(range(len(sizes)))
    if isinstance(spec, dict):
        return [int(i) for i in np.argsort(-np.asarray(sizes), kind='stable')[:spec['largest']]]
    return [int(t) for t in spec]


def nonmember_reference(original, x, y, forget_y):
    """Class-weighted CE of the original on validation nonmembers, weighted like F."""
    weights = np.bincount(forget_y, minlength=2) / len(forget_y)
    x, yt = torch.as_tensor(x, dtype=torch.float32), torch.as_tensor(y, dtype=torch.long)
    return float(sum(w * mean_ce(original, x[y == c], yt[y == c]) for c, w in enumerate(weights) if w > 0))


def ci_low(values, confidence):
    values = np.asarray(values, dtype=float)
    if len(values) < 2:
        return None
    return float(values.mean() - stats.t.ppf(.5 + confidence/2, len(values)-1) * values.std(ddof=1) / np.sqrt(len(values)))


def decide(name, cfg, metrics, noise, retrain_auc_f, retrain_f1, gate, smoke):
    d = lambda key: metrics[key]['distance_F']
    final, matched = f'{name}/final', f'{name}/retain_matched'
    pre = f'{name}/after_kd' if f'{name}/after_kd' in metrics else f'{name}/after_local'
    mg = margin(noise, gate['margin_sd_factor'], gate['minimum_margin'])
    original, candidate = metrics['original'], metrics[final]
    f1 = candidate['utility_test']['macro_f1']
    checks = {
        'identifiable': identifiable(d('original'), noise, gate['identifiability_sd']),
        'mia_positive_control': original['lira_retained_control']['mia_auc'] >= gate['minimum_retained_lira_auc'],
        'local_effect_before_recovery': d('original') - d(pre) > mg,
        'beats_retain_only_matched': d(matched) - d(final) > mg,
        'beats_retain_only_recovery': d('retain_only') - d(final) > mg,
        'utility_floor': f1 >= gate['minimum_macro_f1'],
        'utility_preserved': f1 >= max(original['utility_test']['macro_f1'], retrain_f1) - gate['maximum_f1_drop'],
        'fpr_preserved': candidate['utility_test']['fpr'] <= original['utility_test']['fpr'] + gate['maximum_fpr_increase'],
        'mia_closer_to_retrain': abs(candidate['lira_forget']['mia_auc'] - retrain_auc_f)
                                 <= abs(original['lira_forget']['mia_auc'] - retrain_auc_f),
        'not_oracle_stop': cfg.get('stop_rule', 'none') != 'oracle_retrain'}
    checks = {k: bool(v) for k, v in checks.items()}
    valid = checks['identifiable'] and checks['mia_positive_control']
    return {'method': name, 'checks': checks, 'valid_case': valid, 'margin': mg,
            'distance_original': d('original'), 'distance_pre_recovery': d(pre), 'distance_final': d(final),
            'distance_retain_matched': d(matched), 'distance_retain_only': d('retain_only'),
            'improvement_over_matched': d(matched) - d(final),
            'passed': bool(valid and all(checks.values()) and not smoke)}


def summarize(decisions, config, smoke):
    gate, summary = config['gate'], {}
    for name in config['methods']:
        rows = [d for d in decisions if d['method'] == name]
        valid = [d for d in rows if d['valid_case']]
        excess = [d['improvement_over_matched'] - d['margin'] for d in valid]
        low = ci_low(excess, gate['confidence'])
        rate = float(np.mean([d['passed'] for d in valid])) if valid else None
        oracle = config['methods'][name].get('stop_rule', 'none') == 'oracle_retrain'
        summary[name] = {'cases': len(rows), 'valid_cases': len(valid), 'pass_rate_valid': rate,
                         'mean_excess_over_matched_margin': float(np.mean(excess)) if excess else None,
                         'excess_ci_low': low, 'oracle_stop': oracle,
                         'study_passed': bool(not smoke and not oracle and len(valid) >= gate['minimum_valid_cases']
                                              and rate is not None and rate >= gate['minimum_pass_rate']
                                              and low is not None and low > 0)}
    return summary


def run(data_path, config_path, output, smoke=False):
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    config = json.loads(config_path.read_text())
    if smoke:
        config = merge(config, config.get('smoke', {}))
    torch.set_num_threads(config['threads'])
    torch.use_deterministic_algorithms(True)
    data = load_data(data_path)
    train = dict(data['train'])
    train['type'] = load_types(data_path, train['row_id'])
    if smoke:
        # Explicit smoke-only cap per type, never used as experimental evidence.
        caps = config['cap_per_type']
        keep = np.sort(np.concatenate([np.flatnonzero(train['type'] == t)[:caps.get(t, caps['default'])]
                                       for t in np.unique(train['type'])]))
        train = {k: v[keep] for k, v in train.items()}
    X, y, groups, types = train['X'], train['y'], train['group_id'], train['type']
    test, validation, calibration = data['test'], data['validation'], data['calibration']
    dim, hidden, fl, n_clients = X.shape[1], config['hidden'], config['fl'], config['clients']
    unique_mask = unique_feature_rows(X)
    output.mkdir(parents=True)
    (output/'cases').mkdir()
    started = time.perf_counter()
    manifest = {'config': config, 'smoke_only': smoke, 'data_manifest_sha256': digest(data_path/'manifest.json'),
                'source_hashes': {str(p): digest(p) for root in ['src', 'scripts'] for p in Path(root).rglob('*.py')},
                'python': platform.python_version(), 'torch': torch.__version__, 'numpy': np.__version__,
                'platform': platform.platform(), 'processor': platform.processor(), 'device': 'cpu',
                'timing': 'time.perf_counter; sequential logical workers', 'code_revision': None,
                'train_rows': len(y), 'train_types': pd.Series(types).value_counts().to_dict(),
                'seed_policy': {'retrain_j': f'seed + {RETRAIN_SEED_STRIDE}*j (j=0 is the original init seed)',
                                'shadow_k': f'seed + {SHADOW_SEED_OFFSET} + k',
                                'recovery': f'seed + {RECOVERY_SEED_OFFSET} for every recovery'},
                'compute_matching': 'retain_matched = retain-only local steps equal to FU local minibatch gradient '
                                    'evaluations (forget + retain + KD), then the same recovery',
                'mia': 'offline LiRA, OUT-only FedAvg shadows excluding every query record group; '
                       'thresholds from original on calibration nonmembers only'}
    dump(output/'manifest.json', manifest)

    def workers(parts):
        return [ClientWorker(i, X[idx], y[idx], train['row_id'][idx]) for i, idx in enumerate(parts)]

    originals, decisions, flat, coverage, timing = {}, [], [], [], {}

    def original_for(key, clients, seed):
        if key not in originals:
            t = time.perf_counter()
            m, log = fedavg(model(dim, hidden, seed), workers(clients), fl, fl['rounds'], seed)
            timing[f'original/{"/".join(map(str, key))}'] = time.perf_counter() - t
            originals[key] = (m, predict(m, X))
        return originals[key]

    def run_case(case_id, scenario, seed, clients, target, f, original):
        case_dir = output/'cases'/case_id
        case_dir.mkdir()
        case_start = time.perf_counter()
        retained = retained_partition(clients, target, f)
        retained_ids = np.concatenate(retained)
        if np.intersect1d(retained_ids, f).size:
            raise AssertionError('Forgotten samples in retained training data')
        if any(set(y[idx]) != {0, 1} for idx in retained):
            raise ValueError('A retained client lost a class')
        rw, fw = workers(retained), ClientWorker(target, X[f], y[f], train['row_id'][f])
        dump(case_dir/'request.json', {'scenario': scenario, 'seed': seed, 'target': target,
                                       'forget_row_ids': train['row_id'][f], 'forget_hash': ids_hash(train['row_id'][f]),
                                       'forget_labels': np.bincount(y[f], minlength=2),
                                       'forget_types': pd.Series(types[f]).value_counts().to_dict(),
                                       'client_counts': [len(c) for c in clients],
                                       'retained_counts': [len(c) for c in retained]})
        t = time.perf_counter()
        retrains = []
        for j in range(config['noise_floor']['retrains']):
            s = seed + RETRAIN_SEED_STRIDE * j
            m, _ = fedavg(model(dim, hidden, s), rw, fl, fl['rounds'], s)
            retrains.append(m)
        timing[f'{case_id}/retrains'] = time.perf_counter() - t
        recover = lambda start: fedavg(start, rw, fl, config['recovery_rounds'], seed + RECOVERY_SEED_OFFSET)[0]
        models = {'original': original, 'no_op': copy.deepcopy(original), 'retrain': retrains[0],
                  'retain_only': recover(original)}
        references = {'nonmember': nonmember_reference(original, validation['X'], validation['y'], y[f]),
                      'oracle_retrain': float(np.mean([mean_ce(m, fw.x, fw.y) for m in retrains]))}
        method_info = {}
        for name, cfg in config['methods'].items():
            t = time.perf_counter()
            rule = cfg.get('stop_rule', 'none')
            stop = None if rule == 'none' else references[rule] * cfg.get('stop_scale', 1.)
            local, logs = alternating(original, fw, rw[target], cfg, seed, stop)
            steps, start = gradient_batches(logs), local
            models[f'{name}/after_local'] = local
            kd_steps = 0
            if cfg.get('kd_mode'):
                incompetent = model(dim, hidden, seed + 7) if cfg.get('kd_incompetent') == 'random_init' else None
                kd, losses = distill(original, local, rw[target], cfg, seed, fw, incompetent,
                                     stop if cfg.get('kd_stop') else None)
                kd_steps, start = len(losses), kd
                models[f'{name}/after_kd'] = kd
            models[f'{name}/final'] = recover(start)
            matched, _ = retain_finetune(original, rw[target], steps + kd_steps, cfg, seed)
            models[f'{name}/retain_matched_local'] = matched
            models[f'{name}/retain_matched'] = recover(matched)
            method_info[name] = {'stop_rule': rule, 'stop_target': stop, 'stop_reached': logs[-1]['stop_reached'],
                                 'local_gradient_batches': steps, 'kd_steps': kd_steps}
            dump(case_dir/f'{name}_local_log.json', logs)
            timing[f'{case_id}/{name}'] = time.perf_counter() - t

        # Evaluation sets: F and retained controls (groups disjoint from F) vs class-matched test.
        cap = config['lira'].get('cap_per_class', 2000)
        fm, fn = matched_indices(y[f], test['y'], seed, cap, require_all_classes=False)
        control_pool = retained_ids[~np.isin(groups[retained_ids], groups[f])]
        rm, rn = matched_indices(y[control_pool], test['y'], seed + 2, cap)
        f_eval, c_eval = f[fm], control_pool[rm]
        np.savez_compressed(case_dir/'evaluation_ids.npz', forget=train['row_id'][f_eval],
                            forget_nonmember=test['row_id'][fn], retained=train['row_id'][c_eval],
                            retained_nonmember=test['row_id'][rn])

        # OUT-only shadows: no query record and none of its connected group.
        t = time.perf_counter()
        lira = config['lira']
        pool = np.flatnonzero(~np.isin(groups, np.union1d(groups[f], groups[c_eval])))
        if set(y[pool]) != {0, 1}:
            raise AssertionError('Shadow pool lost a class; lower lira.cap_per_class')
        query_x = np.concatenate([X[f_eval], X[c_eval], test['X'], calibration['X']])
        query_y = np.concatenate([y[f_eval], y[c_eval], test['y'], calibration['y']])
        cuts = np.cumsum([len(f_eval), len(c_eval), len(test['y'])])
        shadow_phi = []
        for k in range(lira['shadows']):
            s = seed + SHADOW_SEED_OFFSET + k
            subset = np.sort(np.random.default_rng(s).choice(pool, int(lira['subset_fraction'] * len(pool)), replace=False))
            parts = [subset[p] for p in partition(y[subset], s, lira.get('alpha', config['alpha']), n_clients)]
            shadow, _ = fedavg(model(dim, hidden, s), workers(parts), fl, lira.get('rounds', fl['rounds']), s)
            shadow_phi.append(logit_confidence(query_y, predict(shadow, query_x)))
        mu, sd = out_statistics(shadow_phi, lira.get('global_variance', True))
        timing[f'{case_id}/shadows'] = time.perf_counter() - t

        def lira_split(m):
            parts = np.split(lira_scores(logit_confidence(query_y, predict(m, query_x)), mu, sd), cuts)
            return dict(zip(['forget', 'control', 'test', 'calibration'], parts))

        original_lira = lira_split(original)
        lira_thresholds = calibrate_thresholds(calibration['y'], original_lira['calibration'])
        loss_thresholds = calibrate_mia(calibration['y'], predict(original, calibration['X']))
        retrain_f = [predict(m, X[f]) for m in retrains]
        retrain_test = [predict(m, test['X']) for m in retrains]
        noise, noise_test = retrain_noise(retrain_f), retrain_noise(retrain_test)

        def evaluate(m, loo=None):
            p_f, p_test = predict(m, X[f]), predict(m, test['X'])
            scores = lira_split(m)
            refs_f = retrain_f if loo is None else retrain_f[:loo] + retrain_f[loo+1:]
            refs_test = retrain_test if loo is None else retrain_test[:loo] + retrain_test[loo+1:]
            return {'distance_F': distance_to_retrains(p_f, refs_f),
                    'distance_test': distance_to_retrains(p_test, refs_test),
                    'utility_test': utility(test['y'], p_test),
                    'utility_validation': utility(validation['y'], predict(m, validation['X'])),
                    'forget_ce': mean_ce(m, fw.x, fw.y),
                    'lira_forget': attack_summary(y[f_eval], scores['forget'], test['y'][fn], scores['test'][fn], lira_thresholds),
                    'lira_retained_control': attack_summary(y[c_eval], scores['control'], test['y'][rn],
                                                            scores['test'][rn], lira_thresholds),
                    'loss_mia_forget': mia(y[f_eval], predict(m, X[f_eval]), test['y'][fn], p_test[fn], loss_thresholds)}

        metrics = {key: evaluate(m, 0 if key == 'retrain' else None) for key, m in models.items()}
        retrain_metrics = [metrics['retrain']] + [evaluate(m, j) for j, m in enumerate(retrains) if j]
        retrain_auc_f = float(np.mean([r['lira_forget']['mia_auc'] for r in retrain_metrics]))
        retrain_f1 = float(np.mean([r['utility_test']['macro_f1'] for r in retrain_metrics]))
        case_decisions = [{'case': case_id, 'scenario': scenario, 'seed': seed, 'target': target,
                           **decide(name, cfg, metrics, noise, retrain_auc_f, retrain_f1, config['gate'], smoke)}
                          for name, cfg in config['methods'].items()]
        dump(case_dir/'metrics.json', {'models': metrics, 'retrain_ensemble': retrain_metrics,
                                       'noise_floor_F': noise, 'noise_floor_test': noise_test,
                                       'stop_references': references, 'methods': method_info,
                                       'lira': {'shadows': lira['shadows'], 'pool_rows': len(pool),
                                                'thresholds': lira_thresholds,
                                                'retrain_mean_forget_auc': retrain_auc_f},
                                       'decisions': case_decisions})
        if config.get('save_checkpoints', True):
            torch.save({'state_dicts': {k: state(m) for k, m in models.items()},
                        'retrains': [state(m) for m in retrains], 'input_dim': dim, 'hidden': hidden},
                       case_dir/'checkpoints.pt')
        for key, r in metrics.items():
            flat.append({'case': case_id, 'scenario': scenario, 'seed': seed, 'target': target, 'forget_rows': len(f),
                         'model': key, 'distance_F': r['distance_F'], 'distance_test': r['distance_test'],
                         'noise_mean_F': noise['mean'], 'noise_sd_F': noise['sd'],
                         'macro_f1': r['utility_test']['macro_f1'], 'fpr': r['utility_test']['fpr'],
                         'forget_ce': r['forget_ce'], 'lira_auc_F': r['lira_forget']['mia_auc'],
                         'lira_auc_control': r['lira_retained_control']['mia_auc'],
                         'loss_mia_auc_F': r['loss_mia_forget']['mia_auc']})
        decisions.extend(case_decisions)
        timing[f'{case_id}/total'] = time.perf_counter() - case_start
        coverage.append({'case': case_id, 'status': 'completed', 'reason': None})
        pd.DataFrame(flat).to_csv(output/'cases.csv', index=False)

    def attempt(case_id, build):
        # Only pre-specified request/partition infeasibility is recorded as invalid; anything else fails.
        try:
            return build()
        except ValueError as error:
            coverage.append({'case': case_id, 'status': 'invalid_request', 'reason': str(error)})
            return None

    try:
        for seed in config['seeds']:
            base = attempt(f'partition_s{seed}', lambda: partition(y, seed, config['alpha'], n_clients))
            if base is None:
                continue
            sizes = [len(c) for c in base]
            for scenario in config['scenarios']:
                kind, sname = scenario['kind'], scenario['name']
                for target in resolve_targets(scenario['targets'], sizes):
                    if kind in ('random', 'atypical'):
                        jobs = [(None, base, ('base', seed))]
                    elif kind == 'family':
                        jobs = []
                        for family in scenario['families']:
                            parts = attempt(f'{sname}_s{seed}_t{target}_{family}', lambda: exclusive_family_partition(
                                y, types, family, target, seed, config['alpha'], n_clients))
                            if parts is not None:
                                jobs.append((family, parts, ('family', seed, family, target)))
                    else:
                        raise ValueError(f'Unknown scenario kind: {kind}')
                    for family, clients, key in jobs:
                        for fraction in scenario['fractions']:
                            case_id = f'{sname}_s{seed}_t{target}' + (f'_{family}' if family else '') + f'_f{fraction}'
                            if kind == 'random':
                                build = lambda: forget_request(y, clients[target], fraction, seed + 1)
                            elif kind == 'family':
                                build = lambda: family_request(types, clients[target], family, fraction, seed + 1)
                            else:
                                train_p = original_for(key, clients, seed)[1]
                                losses = -np.log(np.clip(np.where(y == 1, train_p, 1 - train_p), 1e-7, 1))
                                build = lambda: atypical_request(y, losses, clients[target], fraction, unique_mask)
                            request = attempt(case_id, build)
                            if request is None:
                                continue
                            original = original_for(key, clients, seed)[0]
                            print(f'Case {case_id}: {len(request[0])} forgotten records', flush=True)
                            try:
                                run_case(case_id, sname, seed, clients, target, request[0], original)
                            except ValueError as error:
                                if str(error) != 'A retained client lost a class':
                                    raise
                                coverage.append({'case': case_id, 'status': 'invalid_request', 'reason': str(error)})
        summary = summarize(decisions, config, smoke)
        dump(output/'gate.json', {'methods': summary, 'decisions': decisions, 'smoke_only': smoke,
                                  'study_passed_any': any(v['study_passed'] for v in summary.values()),
                                  'dt_allowed': False,
                                  'note': 'Cases sharing a seed share the original model and are not independent; '
                                          'the CI is a screening statistic, not a robustness claim.'})
    except Exception:
        (output/'error.log').write_text(traceback.format_exc(), encoding='utf-8')
        coverage.append({'case': None, 'status': 'failed', 'reason': 'pipeline'})
        raise
    finally:
        timing['wall_time_total_seconds'] = time.perf_counter() - started
        dump(output/'timings.json', timing)
        pd.DataFrame(coverage, columns=['case', 'status', 'reason']).to_csv(output/'coverage.csv', index=False)
        dump(output/'artifact_hashes.json', {str(p.relative_to(output)): digest(p) for p in output.rglob('*')
                                             if p.is_file() and p.name != 'artifact_hashes.json'})
    print(json.dumps({'output': str(output), 'seconds': timing['wall_time_total_seconds'],
                      'methods': {k: v['study_passed'] for k, v in summary.items()}}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/connected_v3_seed42'))
    parser.add_argument('--config', type=Path, default=Path('configs/study.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    run(args.data, args.config, args.output, args.smoke)
