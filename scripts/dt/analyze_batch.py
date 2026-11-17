"""Summary of the deadline-aware batching runs (direction 3 of docs/dt_extensions.md). Usage:

  python scripts/dt/analyze_batch.py runs/dt_batch results/dt_extensions/batch

ROOT holds one directory per (S, seed, family) written by scripts/dt/run_batch_dt.py. Writes OUT/summary.csv (one row per
family and mode, with the cost relative to `immediate` and the check that every shard_full schedule ends in the same state)
and OUT/report.md. Descriptive only: no gate.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd

from scripts.dt.evaluate_stress import md

COLUMNS = ['family', 'mode', 'budget_level', 'events', 'retrains', 'seconds_total', 'cost_vs_immediate', 'rows_x_rounds',
           'latency_mean', 'latency_max', 'deadline_misses', 'over_budget_events', 'forced_over_budget_events', 'overrun_seconds',
           'served_full_share', 'validation_f1_min', 'validation_f1_final', 'validation_fpr_final', 'exact_all_events', 'final_exact']


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    args = parser.parse_args(argv)
    frames, same = [], []
    for path in sorted(p for p in args.root.iterdir() if (p/'summary.csv').exists()):
        run = json.loads((path/'run.json').read_text())
        frame = pd.read_csv(path/'summary.csv')
        frame.insert(0, 'family', run['family']); frame.insert(1, 'seed', run['seed'])
        if 'immediate' in set(frame['mode']):
            base = frame.set_index('mode').loc['immediate']
            frame['cost_vs_immediate'] = frame.seconds_total/base.seconds_total
            frame['rows_vs_immediate'] = frame.rows_x_rounds/base.rows_x_rounds
        finals = {m: json.loads((path/m/'final.json').read_text()) for m in frame['mode'] if (path/m/'final.json').exists()}
        full_modes = [m for m in ('immediate', 'window5', 'lazy') if m in finals]
        same.append({'family': run['family'], 'seed': run['seed'], 'shard_full_modes': ','.join(full_modes),
                     'final_states_equal': len({json.dumps(finals[m]['weights_sha256'], sort_keys=True) for m in full_modes}) == 1,
                     'bit_exact_checked': all('verify_bit_exact' in f for f in finals.values()),
                     'bit_exact': all(f.get('verify_bit_exact', {}).get('bit_exact', True) for f in finals.values())})
        frames.append(frame)
    if not frames:
        raise SystemExit(f'no runs in {args.root}')
    table = pd.concat(frames, ignore_index=True)
    args.output.mkdir(parents=True, exist_ok=True)
    table.to_csv(args.output/'summary.csv', index=False, float_format='%.6g')
    pd.DataFrame(same).to_csv(args.output/'final_states.csv', index=False)
    shown = table[[c for c in COLUMNS if c in table.columns]].copy()
    for col in ('seconds_total', 'overrun_seconds'):
        shown[col] = shown[col].round(1)
    for col in ('cost_vs_immediate', 'latency_mean', 'latency_max', 'served_full_share', 'validation_f1_min',
                'validation_f1_final', 'validation_fpr_final'):
        if col in shown:
            shown[col] = shown[col].round(4 if 'validation' in col else 3)
    lines = ['# Gom lô và hoãn theo deadline: bảng tổng hợp', '', md(shown, index=False), '',
             '## Trạng thái cuối của các lịch chỉ dùng shard_full (immediate, window5, lazy)', '', md(pd.DataFrame(same), index=False), '']
    (args.output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {(args.output/"report.md").as_posix()}: {len(table)} rows')


if __name__ == '__main__':
    main()
