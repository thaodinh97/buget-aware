"""Summarise the paper-scale runs. Usage: python scripts/shard/analyze_shard_paper_scale.py RESULTS_DIR
(RESULTS_DIR holds paper_a0.1/, paper_a0.5/ and paper_a1.0/ written by run_shard_paper_scale.py)."""
import json, sys, pandas as pd, numpy as np
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
S = sys.argv[1]
A, B, C, M = [], [], [], {}
for a in ['0.1', '0.5', '1.0']:
    A.append(pd.read_csv(f'{S}/paper_a{a}/stream_steps.csv')); B.append(pd.read_csv(f'{S}/paper_a{a}/paper_cases.csv'))
    C.append(pd.read_csv(f'{S}/paper_a{a}/checks.csv')); M[a] = json.load(open(f'{S}/paper_a{a}/meta.json'))
A, B, C = pd.concat(A), pd.concat(B), pd.concat(C)
print('seeds per alpha:', B.groupby('alpha').seed.nunique().to_dict())
for a, m in M.items(): print(a, m.get('scheme', 'per_class'), 'pseed', m['partition_seed'], 'single-class', m['single_class_clients'], 'deleted', round(m['stream_deleted_fraction'], 3))
print('\n== checks ==')
print('Part B affected_shard == full_retrain (bit):', C[C.part == 'B'].affected_shard_equals_full_retrain.all(), len(C[C.part == 'B']))
print('Part A bit-exact:', C[C.part == 'A'].groupby('policy').bit_exact.all().to_dict(), '| per-step exact:', A[A.policy != 'none'].groupby('policy').exact.all().to_dict())
print('\n== Part B (paper protocol), mean over 9 cases x seeds ==')
B['speedup'] = B.T_full/B.seconds.replace(0, np.nan)
g = B.groupby(['alpha', 'method']).agg(Acc=('cal_acc', 'mean'), F1=('cal_macro_f1', 'mean'), F1sd=('cal_macro_f1', 'std'), FPR=('cal_fpr', 'mean'),
    dAcc=('dAcc', 'mean'), dF1=('dF1', 'mean'), dF1max=('dF1', 'max'), dFPR=('dFPR', 'mean'), worst=('client_f1_min', 'mean'), rounds=('rounds', 'mean'),
    secs=('seconds', 'mean'), speedup=('speedup', 'median'))
print(g.round(4).to_string())
print('\n== Part B by contribution group (dynamicfu, utility_stop) ==')
print(B[B.method.isin(['dynamicfu', 'utility_stop', 'rapid'])].groupby(['alpha', 'method', 'group']).agg(contrib=('contribution', 'mean'), rounds=('rounds', 'mean'),
    dF1=('dF1', 'mean'), speedup=('speedup', 'median')).round(3).to_string())
print('\nDynamicFU stop reasons B:', B[B.method == 'dynamicfu'].groupby('alpha').stop_reason.value_counts().to_dict())
print('\n== Part A (stream) final state ==')
ref = A[A.policy == 'full_retrain_single']; refc = ref[ref.step == 10].groupby(['alpha'])[['rows_x_epochs', 'bytes', 'seconds']].mean()
fin = A[((A.step == 10) & (A.policy != 'drop_backfill')) | ((A.policy == 'drop_backfill') & (A.step == 11))]
f = fin.groupby(['alpha', 'policy']).agg(F1=('cal_macro_f1', 'mean'), FPR=('cal_fpr', 'mean'), worst=('client_f1_min', 'mean'), rows=('rows_x_epochs', 'mean'),
    contrib=('contrib_rows', 'mean'), bytes=('bytes', 'mean'), secs=('seconds', 'mean'), rounds=('rounds', 'mean')).reset_index()
f['rows_pct'] = 100*f.rows/f.alpha.map(refc.rows_x_epochs); f['rows+contrib_pct'] = 100*(f.rows + f.contrib.fillna(0))/f.alpha.map(refc.rows_x_epochs)
f['bytes_pct'] = 100*f.bytes/f.alpha.map(refc.bytes); f['time_pct'] = 100*f.secs/f.alpha.map(refc.seconds)
print(f.drop(columns=['rows', 'contrib', 'bytes', 'secs']).round(3).to_string())
print('\n== Part A worst moment during the stream ==')
w = A[A.step <= 10].groupby(['alpha', 'policy']).agg(F1_min=('cal_macro_f1', 'min'), FPR_max=('cal_fpr', 'max'), worst_min=('client_f1_min', 'min'),
    bad=('cal_fpr', lambda s: int((s > .45).sum())))
print(w.round(3).to_string())
r = A[A.policy == 'full_retrain_single'].set_index(['alpha', 'seed', 'step'])
for pol in ['none', 'shard_full', 'shard_fixed', 'dynamicfu', 'utility_stop']:
    x = A[A.policy == pol].set_index(['alpha', 'seed', 'step']); dd = (x.cal_macro_f1 - r.cal_macro_f1).abs(); df = (x.cal_fpr - r.cal_fpr).abs()
    print(f'{pol:13s} vs full retrain per step: mean|dF1| {dd.mean():.4f}  mean|dFPR| {df.mean():.4f}  max|dFPR| {df.max():.4f}')
print('DynamicFU stop reasons A:', A[A.policy == 'dynamicfu'].groupby('alpha').stop_reason.value_counts().to_dict())
