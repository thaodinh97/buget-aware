"""Diagnose where FU's distance gain comes from on finished study runs (default: v7 family cases).

A  decomposition of D(retain-only) - D(FU) into extra local compute on R (retain-only minus
   retain_matched) and the forget signal (retain_matched minus FU);
B  ceiling: could an exact retrain beat retain-only by the noise margin? gap closure per option;
C  per-record direction and magnitude of FU's correction relative to the retrain ensemble;
D  stop-rule accuracy (stop target and pre-recovery CE against retrain CE on F);
E  effect versus strength of F's influence;
P  (optional) whether the client proxy's predictions on F could serve as a per-record target.
Reads checkpoints and metrics only; never retrains the studied models.
"""
import argparse
import json
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy import stats
from src.data.prepare import write_json
from src.data.scenarios import exclusive_family_partition, family_request
from src.eval.noise_floor import distance_to_retrains
from src.fl.fedavg_core import ClientWorker, model, predict
from scripts.run_study import load_types, PROXY_SEED_OFFSET


def ci(values):
    values = np.asarray(values, dtype=float)
    half = stats.t.ppf(.975, len(values) - 1) * values.std(ddof=1) / np.sqrt(len(values))
    return [float(values.mean()), float(values.mean() - half), float(values.mean() + half)]


def cosine(a, b):
    return float(a @ b / np.sqrt((a @ a) * (b @ b))) if (a @ a) > 0 and (b @ b) > 0 else np.nan


def case_rows(runs, method, scenario):
    rows = []
    for run in runs:
        for path in sorted((run/'cases').glob(f'{scenario}_*/metrics.json')):
            m = json.loads(path.read_text())
            dec = next(x for x in m['decisions'] if x['method'] == method and x['candidate'] == 'final')
            if not dec['valid_case'] or dec.get('role', 'evaluation') != 'evaluation':
                continue
            c, noise, info = m['condensed'], m['noise_floor_F'], m['methods'][method]
            rows.append({'run': str(run), 'case': path.parent.name, 'family': m.get('forgotten_type'),
                         'D_orig': c['original']['distance_F'], 'D_retain_only': c['retain_only']['distance_F'],
                         'D_matched': c[f'{method}/retain_matched']['distance_F'], 'D_fu': c[f'{method}/final']['distance_F'],
                         'D_pre': c[f'{method}/after_local']['distance_F'],
                         'noise_mean': noise['mean'], 'noise_sd': noise['sd'], 'margin': dec['margin'],
                         'ce_retrain': float(np.mean([e['forget_ce'] for e in m['retrain_ensemble']])),
                         'ce_pre': m['models'][f'{method}/after_local']['forget_ce'],
                         'stop_target': info['stop_target'], 'stop_reached': info['stop_reached']})
    return pd.DataFrame(rows)


def records(frame, method, train_x, positions):
    out = []
    for _, row in frame.iterrows():
        case_dir = Path(row.run)/'cases'/row.case
        f = np.array([positions[r] for r in json.loads((case_dir/'request.json').read_text())['forget_row_ids']])
        saved = torch.load(case_dir/'checkpoints.pt', weights_only=True)

        def probabilities(state):
            m = model(saved['input_dim'], saved['hidden'], 0)
            m.load_state_dict(state)
            return predict(m, train_x[f]).astype(float)

        sd = saved['state_dicts']
        retrains = np.stack([probabilities(s) for s in saved['retrains']])
        mean, spread = retrains.mean(0), retrains.std(0, ddof=1)
        p_ret, p_fu, p_match = (probabilities(sd[k]) for k in ['retain_only', f'{method}/final', f'{method}/retain_matched'])
        target, move = mean - p_ret, p_fu - p_ret
        off = np.abs(target) > spread  # retain-only is off by more than the retrain spread
        gain = np.abs(p_ret - mean) - np.abs(p_fu - mean)
        top = np.sort(gain)[::-1][:max(1, len(f) // 10)]
        out.append({'case': row.case, 'cos_fu_move_to_target': cosine(move, target),
                    'cos_compute_to_target': cosine(p_match - p_ret, target),
                    'cos_forget_to_target': cosine(p_fu - p_match, target),
                    'share_records_off': float(off.mean()),
                    'share_right_direction_off': float(np.mean(np.sign(move[off]) == np.sign(target[off]))) if off.any() else np.nan,
                    'share_overshoot_off': float(np.mean(np.sign(p_fu - mean)[off] != np.sign(p_ret - mean)[off])) if off.any() else np.nan,
                    'correction_fraction': float(np.linalg.norm(move) / np.linalg.norm(target)),
                    'top10_share_of_gain': float(top.sum() / gain.sum()) if gain.sum() > 0 else np.nan,
                    'alert_original': float((probabilities(sd['original']) >= .5).mean()),
                    'alert_retrain_mean': float((mean >= .5).mean())})
    return pd.DataFrame(out)


def proxy_check(frame, train, types, config):
    out = []
    x, y, fl = train['X'], train['y'], config['fl']
    for _, row in frame.iterrows():
        seed, target, family = re.match(r'family_s(\d+)_t(\d+)_(\w+)_f1', row.case).groups()
        seed, target = int(seed), int(target)
        clients = exclusive_family_partition(y, types, family, target, seed, config['alpha'], config['clients'])
        f, r = family_request(types, clients[target], family, 1.0, seed + 1)
        s = seed + PROXY_SEED_OFFSET
        state, _, _ = ClientWorker(target, x[r], y[r], train['row_id'][r]).train(
            model(x.shape[1], config['hidden'], s),
            {'lr': fl['lr'], 'batch_size': fl['batch_size'], 'local_epochs': config['proxy']['epochs']}, s)
        proxy = model(x.shape[1], config['hidden'], s)
        proxy.load_state_dict(state)
        saved = torch.load(Path(row.run)/'cases'/row.case/'checkpoints.pt', weights_only=True)

        def probabilities(st):
            m = model(saved['input_dim'], saved['hidden'], 0)
            m.load_state_dict(st)
            return predict(m, x[f]).astype(float)

        retrains = [probabilities(st) for st in saved['retrains']]
        out.append({'case': row.case, 'family': family, 'D_proxy': distance_to_retrains(predict(proxy, x[f]), retrains),
                    'D_retain_only_rep0': distance_to_retrains(probabilities(saved['state_dicts']['retain_only']), retrains)})
    return pd.DataFrame(out)


def investigate(runs, data, output, method, scenario, with_proxy):
    output.mkdir(parents=True, exist_ok=True)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    frame = case_rows(runs, method, scenario)
    frame['gain_compute'] = frame.D_retain_only - frame.D_matched
    frame['gain_forget'] = frame.D_matched - frame.D_fu
    frame['gain_total'] = frame.D_retain_only - frame.D_fu
    frame['retrain_residual'] = frame.D_retain_only - frame.noise_mean
    gap = frame.D_orig - frame.noise_mean
    summary = {'cases': len(frame), 'method': method, 'scenario': scenario,
               'A_gain_total': ci(frame.gain_total), 'A_gain_extra_local_compute': ci(frame.gain_compute),
               'A_gain_forget_signal': ci(frame.gain_forget),
               'A_share_from_extra_local_compute': float(frame.gain_compute.sum() / frame.gain_total.sum()),
               'B_share_retrain_beats_retain_only_by_margin': float((frame.retrain_residual > frame.margin).mean()),
               'B_share_fu_beats_retain_only_by_margin': float((frame.gain_total > frame.margin).mean()),
               'B_gap_closure_median': {k: float(((frame.D_orig - frame[c]) / gap).median())
                                        for k, c in [('retain_only', 'D_retain_only'), ('retain_matched', 'D_matched'),
                                                     ('fu_final', 'D_fu'), ('fu_pre_recovery', 'D_pre')]},
               'D_median_abs_log_target_error': float(np.abs(np.log(frame.stop_target / frame.ce_retrain)).median()),
               'D_share_target_below_retrain_ce': float((frame.stop_target < frame.ce_retrain).mean()),
               'D_share_pre_recovery_ce_above_retrain': float((frame.ce_pre > frame.ce_retrain).mean()),
               'D_share_stop_reached': float(frame.stop_reached.mean()),
               'E_spearman_signal_vs_effect': list(stats.spearmanr((frame.D_orig - frame.noise_mean) / frame.noise_sd,
                                                                  frame.gain_total / frame.margin))}
    data_arrays = dict(np.load(data/'train.npz', allow_pickle=False))
    rec = records(frame, method, data_arrays['X'], {r: i for i, r in enumerate(data_arrays['row_id'])})
    summary['C_records_median'] = rec.drop(columns='case').median().to_dict()
    frame.to_csv(output/'cases.csv', index=False)
    rec.to_csv(output/'records.csv', index=False)
    if with_proxy:
        config = json.loads((Path(runs[0])/'manifest.json').read_text())['config']
        prox = proxy_check(frame, data_arrays, load_types(data, data_arrays['row_id']), config)
        prox.to_csv(output/'proxy.csv', index=False)
        summary['P_proxy'] = {'D_proxy_mean': float(prox.D_proxy.mean()),
                              'D_retain_only_mean': float(prox.D_retain_only_rep0.mean()),
                              'cases_proxy_closer': int((prox.D_proxy < prox.D_retain_only_rep0).sum())}
    write_json(output/'summary.json', summary)
    return summary


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--runs', type=Path, nargs='+', default=[Path('runs/study_v7_a'), Path('runs/study_v7_b')])
    parser.add_argument('--data', type=Path, default=Path('data/processed/connected_log1p_v5_seed42'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--method', default='neggrad_plus_proxy')
    parser.add_argument('--scenario', default='family')
    parser.add_argument('--proxy', action='store_true')
    args = parser.parse_args()
    print(json.dumps(investigate(args.runs, args.data, args.output, args.method, args.scenario, args.proxy), indent=2))
