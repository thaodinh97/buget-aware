"""Summarise replays of the v2 stream through the component (scripts/shard/replay_v2_stream.py). Usage:
python scripts/shard/analyze_replay.py REPLAY_ROOT [--reference V2_DIR_THIS_MACHINE] [--stored results/shard_v2/stream_20c]

REPLAY_ROOT holds one directory per (S, seed) written by replay_v2_stream.py. Sections:
1. equivalence       compare.json of every replay (component against the v2 run it was compared with)
2. machine           the v2 run made on this machine (--reference) against the stored v2 run (--stored), same seeds:
                     where round counts differ and how far the calibration metrics are apart
3. cost              state after the 10 requests per S and policy, mean over the replayed seeds; with --reference also as a
                     share of retraining a single model after every request
4. client compute    seconds of one local round against the client's samples (least squares over all training records)
Descriptive only: no gate, no test split.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

pd.set_option('display.width', 250); pd.set_option('display.max_columns', 40)
HIGH_FPR = .45
KEY = ['S', 'seed', 'policy', 'step']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('--reference', type=Path)
    parser.add_argument('--stored', type=Path)
    args = parser.parse_args(argv)
    runs = sorted(p for p in args.root.iterdir() if (p/'steps.csv').exists() and (p/'run.json').exists())
    if not runs:
        raise SystemExit(f'no replay directories in {args.root}')
    steps = pd.concat([pd.read_csv(p/'steps.csv') for p in runs], ignore_index=True)
    meta = [json.loads((p/'run.json').read_text()) for p in runs]
    seeds = sorted(int(s) for s in steps.seed.unique()); machine = meta[0]['machine']
    print(f'{len(runs)} replays: S {sorted(int(s) for s in steps.S.unique())}, seeds {seeds}; data is the v2 data: {all(m["data_is_v2"] for m in meta)}')
    print(f'machine: {machine["platform"]}, {machine["cpu_count"]} CPUs, torch {machine["torch"]}, {machine["threads"]} threads, {machine["device"]}')

    print('\n== 1. Component against the v2 run it was compared with ==')
    rows = []
    for p, m in zip(runs, meta):
        if not (p/'compare.json').exists():
            continue
        c = json.loads((p/'compare.json').read_text())
        for policy, v in c['policies'].items():
            rows.append({'S': m['shards'], 'seed': m['seed'], 'policy': policy, 'reference': c.get('reference'),
                         'steps': v['steps_compared'], 'discrete_mismatches': v['mismatches'],
                         'metrics_max_abs_diff': v['metrics_max_abs_diff'], 'bit_exact_end': m['end'][policy]['bit_exact']})
    if rows:
        eq = pd.DataFrame(rows)
        print('compared with:', sorted(eq.reference.dropna().unique()))
        print(eq.groupby('policy', sort=False).agg(replays=('steps', 'size'), steps=('steps', 'sum'), discrete_mismatches=('discrete_mismatches', 'sum'),
              metrics_max_abs_diff=('metrics_max_abs_diff', 'max'), bit_exact_end=('bit_exact_end', 'all')).to_string())
        print(f'all equal: {bool((eq.discrete_mismatches == 0).all() and (eq.metrics_max_abs_diff == 0).all())}; '
              f'every included shard bit-equal to a fresh retrain at the end: {bool(eq.bit_exact_end.all())}')
    else:
        print('no compare.json found')
    exact = steps.groupby('policy', sort=False)[['exact', 'structure_ok']].all()
    print('exact and structure_ok at every step:', bool(exact.all().all()))

    if args.reference and args.stored:
        print('\n== 2. v2 runner on this machine against the stored v2 run (same data, seeds and code) ==')
        here = pd.read_csv(args.reference/'steps.csv'); there = pd.read_csv(args.stored/'steps.csv')
        both = here.merge(there, on=KEY, suffixes=('', '_stored'))
        both = both[(both.policy != 'none') & (both.step <= 10)]
        both['rounds_differ'] = both.rounds != both.rounds_stored
        both['dF1'] = (both.cal_macro_f1 - both.cal_macro_f1_stored).abs(); both['dFPR'] = (both.cal_fpr - both.cal_fpr_stored).abs()
        both['high_here'] = both.cal_fpr > HIGH_FPR; both['high_stored'] = both.cal_fpr_stored > HIGH_FPR
        print(f'seeds {sorted(int(s) for s in both.seed.unique())}; rows are (seed, step)')
        print(both.groupby(['S', 'policy']).agg(rows=('step', 'size'), rounds_differ=('rounds_differ', 'sum'), dF1_max=('dF1', 'max'),
              dFPR_max=('dFPR', 'max'), fpr_gt_045_here=('high_here', 'sum'), fpr_gt_045_stored=('high_stored', 'sum')).round(4).to_string())
        fin = both[both.step == 10]
        print('\nfinal state (step 10), mean over seeds: this machine / stored')
        print(fin.groupby(['S', 'policy']).agg(F1=('cal_macro_f1', 'mean'), F1_stored=('cal_macro_f1_stored', 'mean'), FPR=('cal_fpr', 'mean'),
              FPR_stored=('cal_fpr_stored', 'mean'), rounds=('rounds', 'mean'), rounds_stored=('rounds_stored', 'mean')).round(4).to_string())
        base = here[here.policy == 'none'].merge(there[there.policy == 'none'], on=KEY, suffixes=('', '_stored'))
        base = base[base.step == 1]
        print('\ninitial ensemble (policy none), |difference| max over seeds:',
              base.assign(d=(base.cal_macro_f1 - base.cal_macro_f1_stored).abs()).groupby('S').d.max().round(5).to_dict())

    print('\n== 3. Cost after the 10 requests (drop_backfill: after its backlog), mean over seeds ==')
    last = steps.groupby(['S', 'seed', 'policy'], sort=False).tail(1)
    sums = steps.groupby(['S', 'seed', 'policy'], sort=False)[['client_rounds', 'bytes_up_raw', 'bytes_down_raw', 'bytes_up_packed', 'bytes_down_packed',
        'compute_seconds_sum', 'server_aggregate_seconds', 'policy_eval_seconds', 'makespan_seconds_simulated', 'wall_seconds_total']].sum().reset_index()
    tab = last[['S', 'seed', 'policy', 'rounds', 'rows_x_rounds', 'cal_macro_f1', 'cal_fpr']].merge(sums, on=['S', 'seed', 'policy'])
    tab['bytes_packed'] = tab.bytes_up_packed + tab.bytes_down_packed; tab['bytes_raw'] = tab.bytes_up_raw + tab.bytes_down_raw
    cols = ['rounds', 'client_rounds', 'rows_x_rounds', 'bytes_raw', 'bytes_packed', 'compute_seconds_sum', 'makespan_seconds_simulated',
            'policy_eval_seconds', 'wall_seconds_total', 'cal_macro_f1', 'cal_fpr']
    cost = tab.groupby(['S', 'policy'], sort=False)[cols].mean()
    if args.reference:
        ref = pd.read_csv(args.reference/'steps.csv')
        ref = ref[(ref.policy == 'full_retrain_single') & (ref.step == 10) & ref.seed.isin(seeds)]
        if len(ref):
            cost.insert(3, 'rows_pct_of_single_retrain', 100*cost.rows_x_rounds/ref.rows_x_rounds.mean())
    print(cost.round(3).to_string())
    worst = steps[steps.step <= 10].groupby(['S', 'policy'], sort=False).agg(makespan_max=('makespan_seconds_simulated', 'max'),
        makespan_median=('makespan_seconds_simulated', 'median'), clients_max=('client_rounds', 'max'), F1_min=('cal_macro_f1', 'min'),
        FPR_max=('cal_fpr', 'max'), fpr_gt_045=('cal_fpr', lambda v: int((v > HIGH_FPR).sum())), rss_peak_mb=('rss_bytes_peak_sampled', lambda v: v.max()/2**20))
    print('\nper request: slowest and median simulated makespan (s), worst utility during the stream')
    print(worst.round(3).to_string())
    fit = pd.DataFrame([{'S': m['shards'], 'seed': m['seed'], **{k: m['fit']['cost'][k] for k in ('rounds', 'client_rounds', 'rows_x_rounds',
        'bytes_up_packed', 'bytes_down_packed', 'compute_seconds_sum', 'makespan_seconds_simulated', 'makespan_parallel_shards_seconds_simulated',
        'wall_seconds_total', 'models_stored', 'state_bytes_in_memory')}, 'inference_ms_per_1k_rows': 1000*m['fit']['inference']['ensemble_seconds_per_1k_rows'],
        'single_shard_ms_per_1k_rows': 1e6*m['fit']['inference']['single_shard_seconds']/m['fit']['inference']['rows']} for m in meta])
    print('\nbase ensemble (fit), mean over seeds')
    print(fit.groupby('S').mean(numeric_only=True).drop(columns='seed').round(3).to_string())

    print('\n== 4. Client compute time of one local round against its samples (training records of every replay) ==')
    clients = pd.concat([pd.read_csv(p/'client_costs.csv') for p in runs if (p/'client_costs.csv').exists()], ignore_index=True)
    clients = clients[clients.phase != 'contribution']
    x, y = clients.samples.to_numpy(float), clients.compute_seconds.to_numpy(float)
    slope, intercept = np.polyfit(x, y, 1); pred = intercept + slope*x
    r2 = 1 - ((y - pred)**2).sum()/((y - y.mean())**2).sum()
    print(f'{len(clients)} records, {clients.client_id.nunique()} clients, samples {int(x.min())}..{int(x.max())}')
    print(f'seconds = {intercept*1000:.2f} ms + {slope*1e6:.2f} ms per 1000 samples; R^2 = {r2:.3f}; '
          f'median |relative error| = {np.median(np.abs(pred - y)/y):.3f}')
    packed = clients.bytes_up_packed.dropna()
    print(f'payload per client and round: raw {int(clients.bytes_up_raw.iloc[0])} bytes, packed up {int(packed.min())}..{int(packed.max())} bytes')


if __name__ == '__main__':
    main()
