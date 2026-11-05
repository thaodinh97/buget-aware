"""Summarise the two 20-client runs. Usage:
python scripts/shard/analyze_shard_20c.py ENSEMBLE_DIR STREAM_DIR
(ENSEMBLE_DIR from run_shard_ensemble_check.py, STREAM_DIR from run_shard_stream.py)."""
import sys
import pandas as pd
pd.set_option('display.width', 260); pd.set_option('display.max_columns', 40)
E, ST = sys.argv[1], sys.argv[2]

print('== 1. Ensemble utility by number of shards S (calibration split, 5 seeds) ==')
e = pd.read_csv(f'{E}/ensemble.csv')
print(e.groupby('S')[['cal_macro_f1', 'cal_fpr', 'client_f1_min']].agg(['mean', 'std']).round(4).to_string())
b = e[e.S == 1].set_index('seed')
for S in sorted(e.S.unique())[1:]:
    d = e[e.S == S].set_index('seed').cal_macro_f1 - b.cal_macro_f1
    print(f'S={S}: macro-F1 minus S=1, mean {d.mean():+.4f} [{d.min():+.4f}, {d.max():+.4f}]')

print('\n== 2. Exact unlearning of one request by retraining the affected shard for R rounds ==')
u = pd.read_csv(f'{E}/unlearning.csv')
u['rows_frac'] = u.rows_x_rounds/u.full_retrain_rows_x_rounds; u['bytes_frac'] = u.bytes/u.full_retrain_bytes
print(u.groupby(['request', 'S', 'rounds']).agg(F1=('cal_macro_f1', 'mean'), FPR=('cal_fpr', 'mean'), worst=('client_f1_min', 'mean'),
      rows_frac=('rows_frac', 'mean'), bytes_frac=('bytes_frac', 'mean')).round(4).to_string())

print('\n== 3. Sequential stream of 10 requests ==')
s = pd.read_csv(f'{ST}/steps.csv'); x = pd.read_csv(f'{ST}/exactness.csv')
print('bit-exact at the end:', x.groupby(['S', 'policy']).bit_exact.all().to_dict())
print('exact at every step:', s[s.policy != 'none'].groupby(['S', 'policy']).exact.all().to_dict())
ref = s[(s.policy == 'full_retrain_single') & (s.step == 10)][['rows_x_rounds', 'bytes', 'seconds']].mean()
fin = s[((s.step == 10) & (s.policy != 'drop_backfill')) | ((s.policy == 'drop_backfill') & (s.step == 11))]
g = fin.groupby(['S', 'policy']).agg(F1=('cal_macro_f1', 'mean'), FPR=('cal_fpr', 'mean'), worst=('client_f1_min', 'mean'),
    rows=('rows_x_rounds', 'mean'), bytes=('bytes', 'mean'), secs=('seconds', 'mean'))
g['rows_pct'] = 100*g.rows/ref.rows_x_rounds; g['bytes_pct'] = 100*g.bytes/ref.bytes; g['time_pct'] = 100*g.secs/ref.seconds
print(g.drop(columns=['rows', 'bytes', 'secs']).round(3).to_string())
w = s[s.step <= 10].groupby(['S', 'policy']).agg(F1_min=('cal_macro_f1', 'min'), FPR_max=('cal_fpr', 'max'), worst_min=('client_f1_min', 'min'),
    bad=('cal_fpr', lambda v: int((v > .45).sum())))
print('\nworst moment during the stream:\n' + w.round(3).to_string())
if 'stop_reason' in s: print('\nDynamicFU stop reasons:', s[s.policy == 'dynamicfu'].groupby('S').stop_reason.value_counts().to_dict())
