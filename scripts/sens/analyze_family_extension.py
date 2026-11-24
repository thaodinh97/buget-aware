"""Descriptive analysis of a stream-family extension of C1/C2 (docs/dt_concentrated_plan.md), side by side with the
families of the C1/C2 run. No hypothesis tests.

    python scripts/sens/analyze_family_extension.py results/dt_concentrated

Writes <run>/analysis/{arms.csv, diffs.csv, report.md}. Paired = same (seed, family, load, environment) block; the
extension and C1/C2 share seeds, arrivals, request sizes and stress paths, so blocks of different families with the same
(seed, load, environment) differ only in where the requests land.
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
C1C2 = ROOT/'results/dt_orchestration/summary.csv'
KEY = ['seed', 'family', 'load', 'environment']
Y = 'joint_success_requests'
B, SEED = 10000, 1


def ci(x):
    x = np.asarray(x, float)
    if not len(x):
        return np.nan, np.nan, np.nan
    means = x[np.random.default_rng(SEED).integers(0, len(x), (B, len(x)))].mean(1)
    return float(x.mean()), *map(float, np.percentile(means, [2.5, 97.5]))


def fmt(t, digits=2):
    return '–' if np.isnan(t[0]) else f'{t[0]:+.{digits}f} [{t[1]:+.{digits}f}; {t[2]:+.{digits}f}]'.replace('.', ',')


def num(v, digits=0):
    return f'{v:,.{digits}f}'.replace(',', ' ').replace('.', ',').replace(' ', '.')


def arms_table(df):
    g = df.groupby(['family', 'predictor']).agg(blocks=('block', 'nunique'), requests=('requests', 'sum'), joint=(Y, 'sum'),
                                                ontime=('deadline_success_rate', 'mean'),
                                                within=('within_budget_requests', 'sum'), retrains=('events', 'sum'),
                                                full=('full_requests', 'sum'), p95=('latency_p95_seconds', 'mean'),
                                                f1=('calibration_macro_f1', 'mean'), fpr=('calibration_fpr', 'mean'),
                                                exact=('bit_exact', 'all'), exact_events=('exact_all_events', 'all'))
    return g.reset_index()


def diffs(df, arms):
    rows = []
    for fam, sub in df.groupby('family'):
        wide = sub.pivot_table(index=KEY, columns='predictor', values=Y)
        for other in arms:
            if other == 'dt_queue' or other not in wide:
                continue
            d = (wide['dt_queue'] - wide[other]).dropna()
            for scope, part in [('all', d)] + [(f'load={l:g}', d.xs(l, level='load', drop_level=False)) for l in
                                                sorted(d.index.get_level_values('load').unique())] \
                    + [(f'env={e}', d.xs(e, level='environment', drop_level=False)) for e in
                       sorted(d.index.get_level_values('environment').unique())]:
                m, lo, hi = ci(part.to_numpy())
                rows.append(dict(family=fam, comparison=f'dt_queue - {other}', scope=scope, n=len(part), mean=m, lo=lo,
                                 hi=hi, win=int((part > 0).sum()), tie=int((part == 0).sum()), loss=int((part < 0).sum())))
    return pd.DataFrame(rows)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('run', type=Path)
    args = parser.parse_args(argv)
    run = args.run
    cfg = json.loads((run/'protocol_lock.json').read_text(encoding='utf-8'))['config']
    arms = cfg['arms']
    ext = pd.read_csv(run/'summary.csv')
    ref = pd.read_csv(C1C2)
    ref = ref[ref.predictor.isin(arms) & ref.seed.isin(cfg['seeds'])]
    both = pd.concat([ext, ref], ignore_index=True)
    out = run/'analysis'
    out.mkdir(exist_ok=True)
    a, d = arms_table(both), diffs(both, arms)
    a.to_csv(out/'arms.csv', index=False)
    d.to_csv(out/'diffs.csv', index=False)
    fams = [f for f in cfg['families']] + [f for f in ('spread', 'proportional') if f in set(ref.family)]
    complete = (run/'complete.json').exists()
    nblocks = ext.block.nunique()
    lines = [f'# Mở rộng C1/C2 sang họ {", ".join(cfg["families"])}: kết quả mô tả', '',
             f'Sinh bởi `scripts/sens/analyze_family_extension.py`. Protocol: `docs/dt_concentrated_plan.md`. '
             f'Block của phần mở rộng: {nblocks}{"" if complete else " (**chưa đủ**)"}; cột `spread` và `proportional` lấy từ C1/C2 '
             '(cùng seed). Chênh lệch = trung bình theo block [khoảng tin cậy bootstrap 95 %], đơn vị yêu cầu/block (trên 12). '
             'Không kiểm định giả thuyết.', '',
             '## Theo arm', '',
             '| Họ | Arm | Thành công đồng thời | Đúng hạn | Trong ngân sách | Số lần train lại | Chạy full | p95 độ trễ (s) | Macro-F1 | FPR | Chính xác |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---|']
    for fam in fams:
        for arm in arms:
            r = a[(a.family == fam) & (a.predictor == arm)]
            if not len(r):
                continue
            r = r.iloc[0]
            lines.append(f"| `{fam}` | `{arm}` | {int(r.joint)}/{int(r.requests)} ({num(100*r.joint/r.requests, 1)} %) | "
                         f"{num(100*r.ontime, 1)} % | {int(r.within)} | {int(r.retrains)} | {int(r.full)} | {num(r.p95, 2)} | "
                         f"{num(r.f1, 4)} | {num(r.fpr, 4)} | {'có' if r.exact and r.exact_events else '**không**'} |")
    for scope_title, prefix in (('Toàn bộ', 'all'), ('Theo tải', 'load='), ('Theo môi trường', 'env=')):
        lines += ['', f'## `dt_queue` − arm khác: {scope_title.lower()}', '',
                  '| So sánh | Phạm vi | ' + ' | '.join(f'`{f}`' for f in fams) + ' |', '|---|---|' + '---:|'*len(fams)]
        scopes = sorted(s for s in d.scope.unique() if s.startswith(prefix)) if prefix != 'all' else ['all']
        for other in [x for x in arms if x != 'dt_queue']:
            for s in scopes:
                cells = []
                for fam in fams:
                    r = d[(d.family == fam) & (d.comparison == f'dt_queue - {other}') & (d.scope == s)]
                    cells.append('–' if not len(r) else f"{fmt((r.iloc[0]['mean'], r.iloc[0]['lo'], r.iloc[0]['hi']))} "
                                 f"({int(r.iloc[0].win)}/{int(r.iloc[0].tie)}/{int(r.iloc[0].loss)})")
                lines.append(f'| dt_queue − `{other}` | {s} | ' + ' | '.join(cells) + ' |')
    lines += ['', 'Trong ngoặc: số block thắng / hòa / thua của `dt_queue`.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'{nblocks} blocks; wrote {out/"report.md"}')


if __name__ == '__main__':
    main()
