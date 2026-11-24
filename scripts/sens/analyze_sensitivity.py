"""Descriptive analysis of the sensitivity sweep (docs/dt_sensitivity_plan.md). No hypothesis tests.

    python scripts/sens/analyze_sensitivity.py          # writes results/dt_sensitivity/analysis/{cells.csv, arms.csv, report.md}

Reads results/dt_sensitivity/<config>/summary.csv for every config that has one (an unfinished config is analysed on the
blocks it has and flagged) and, as the reference of S2, S3 and S5 and as the comparison for the S1 rerun, the matching
blocks of the C1/C2 run (results/dt_orchestration/summary.csv). Paired = same (seed, family, load, environment).
"""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd

from scripts.sens.run_sensitivity import CONFIG_DIR, OUT, stages

C1C2 = ROOT/'results/dt_orchestration/summary.csv'
KEY = ['seed', 'family', 'load', 'environment']
Y = 'joint_success_requests'
RESAMPLES, BOOT_SEED = 10000, 1
TWIN_PARAMS = {'q': ('margin_q', ['dt_queue', 'rows_legacy']), 'w': ('margin_window', ['dt_queue', 'rows_legacy']),
               'eta': ('expert_eta', ['dt_queue']), 'floor': ('expert_floor', ['dt_queue']),
               'alpha': ('alpha', ['rows_legacy'])}


def ci(x):
    x = np.asarray(x, float)
    if len(x) == 0:
        return np.nan, np.nan, np.nan
    rng = np.random.default_rng(BOOT_SEED)
    means = x[rng.integers(0, len(x), (RESAMPLES, len(x)))].mean(1)
    return x.mean(), *np.percentile(means, [2.5, 97.5])


def paired(a_rows, b_rows, value=Y):
    """a - b per block; a_rows, b_rows: rows of one arm each."""
    m = a_rows.set_index(KEY)[value].sub(b_rows.set_index(KEY)[value]).dropna()
    return m.to_numpy()


def describe(diff):
    mean, lo, hi = ci(diff)
    return dict(n=len(diff), mean=mean, lo=lo, hi=hi, win=int((diff > 0).sum()), tie=int((diff == 0).sum()),
                loss=int((diff < 0).sum()))


def fmt(d, digits=2):
    if not d or d['n'] == 0:
        return '–'
    return f"{d['mean']:+.{digits}f} [{d['lo']:+.{digits}f}, {d['hi']:+.{digits}f}]"


def load_configs():
    out = {}
    for name, (stage, _) in stages().items():
        path = OUT/name/'summary.csv'
        if path.exists():
            cfg = json.loads((CONFIG_DIR/f'{name}.json').read_text(encoding='utf-8'))
            df = pd.read_csv(path)
            out[name] = dict(stage=stage, cfg=cfg, df=df, complete=(OUT/name/'complete.json').exists(),
                             blocks=df['block'].nunique())
    return out


def reference(cfg, c1c2, blocks=None):
    r = c1c2[c1c2.seed.isin(cfg['seeds']) & c1c2.load.isin(cfg['loads']) & c1c2.environment.isin(cfg['environments'])
             & c1c2.family.isin(cfg['families']) & c1c2.predictor.isin(cfg['arms'])]
    if blocks is not None:     # only the blocks the config has (an unfinished config)
        r = r.merge(blocks, on=KEY)
    return r


def arm(df, name):
    return df[df.predictor == name]


def comparisons(df, arms, scope):
    rows = []
    for other in arms:
        if other == 'dt_queue':
            continue
        d = describe(paired(arm(df, 'dt_queue'), arm(df, other)))
        u = describe(paired(arm(df, 'dt_queue'), arm(df, other), 'calibration_macro_f1'))
        rows.append(dict(scope, comparison=f'dt_queue - {other}', **d, f1_mean=u['mean'], f1_lo=u['lo'], f1_hi=u['hi']))
    return rows


def arm_table(df, scope):
    g = df.groupby('predictor').agg(blocks=('block', 'nunique'), requests=('requests', 'mean'), joint=(Y, 'mean'),
                                    deadline_rate=('deadline_success_rate', 'mean'),
                                    within_budget=('within_budget_requests', 'mean'),
                                    calibration_f1=('calibration_macro_f1', 'mean'),
                                    exact=('bit_exact', 'all'), exact_events=('exact_all_events', 'all'))
    return [dict(scope, arm=k, **v) for k, v in g.to_dict('index').items()]


def main():
    configs = load_configs()
    c1c2 = pd.read_csv(C1C2)
    analysis = OUT/'analysis'
    analysis.mkdir(parents=True, exist_ok=True)
    cells, arms_rows, lines = [], [], []
    total = len(stages())
    done = sum(c['complete'] for c in configs.values())
    lines += ['# Phân tích độ nhạy: kết quả mô tả', '',
              f'Sinh bởi `scripts/sens/analyze_sensitivity.py`. Protocol: [docs/dt_sensitivity_plan.md](../../../docs/dt_sensitivity_plan.md). '
              f'Config hoàn tất: **{done}/{total}**; có dữ liệu: {len(configs)}. '
              'Ô = trung bình chênh lệch ghép cặp theo block [khoảng tin cậy bootstrap 95 %]; đơn vị: yêu cầu vừa đúng hạn vừa '
              'trong ngân sách mỗi block. Không có kiểm định giả thuyết.', '']
    unfinished = [n for n, c in configs.items() if not c['complete']]
    if unfinished:
        lines += [f'Chưa xong (phân tích trên số block đã có): {", ".join(f"`{n}` ({configs[n]["blocks"]} block)" for n in unfinished)}', '']
    for name, c in configs.items():
        df, cfg = c['df'], c['cfg']
        scope = dict(config=name, stage=c['stage'], complete=c['complete'], subset='all')
        cells += comparisons(df, cfg['arms'], scope)
        arms_rows += arm_table(df, scope)
        for env, sub in df.groupby('environment'):
            cells += comparisons(sub, cfg['arms'], dict(scope, subset=f'env={env}'))
        for load, sub in df.groupby('load'):
            cells += comparisons(sub, cfg['arms'], dict(scope, subset=f'load={load:g}'))
    cells_df, arms_df = pd.DataFrame(cells), pd.DataFrame(arms_rows)
    cells_df.to_csv(analysis/'cells.csv', index=False)
    arms_df.to_csv(analysis/'arms.csv', index=False)
    if len(arms_df):
        bad = arms_df[~(arms_df.exact & arms_df.exact_events)]
        lines += [f'**Tính chính xác:** {len(arms_df) - len(bad)}/{len(arms_df)} (config, arm) unlearn chính xác và trùng từng bit.', '']

    def cell(name, comparison, subset='all'):
        if name not in configs:
            return None
        r = cells_df[(cells_df.config == name) & (cells_df.comparison == comparison) & (cells_df.subset == subset)]
        return r.iloc[0].to_dict() if len(r) else None

    # S1: B x D grids
    bs, ds = (0.5, 0.75, 1.0, 1.5), (1.0, 1.5, 2.0, 3.0)
    lines += ['## S1: ngân sách B × deadline D (tải 0,2 F, seed 73042–73043, cả 3 môi trường)', '']
    for comparison in ('dt_queue - edf', 'dt_queue - immediate', 'dt_queue - rows_legacy'):
        for subset in ('all', 'env=none', 'env=gradual_clients', 'env=jitter_bandwidth'):
            lines += [f'**{comparison}**, {subset}', '', '| B \\ D | ' + ' | '.join(f'{d:g} F' for d in ds) + ' |',
                      '|---|' + '---:|'*len(ds)]
            for b in bs:
                lines.append(f'| {b:g} F | ' + ' | '.join(fmt(cell(f's1_b{b:g}_d{d:g}', comparison, subset)) for d in ds) + ' |')
            lines.append('')
    lines += ['**Trung bình `dt_queue`** (yêu cầu/block, trên 12)', '', '| B \\ D | ' + ' | '.join(f'{d:g} F' for d in ds) + ' |',
              '|---|' + '---:|'*len(ds)]
    for b in bs:
        vals = []
        for d in ds:
            r = arms_df[(arms_df.config == f's1_b{b:g}_d{d:g}') & (arms_df.arm == 'dt_queue')] if len(arms_df) else []
            vals.append(f"{r.iloc[0]['joint']:.2f}" if len(r) else '–')
        lines.append(f'| {b:g} F | ' + ' | '.join(vals) + ' |')
    lines.append('')
    ref = 's1_b0.75_d1.5'
    if ref in configs:
        df = configs[ref]['df']
        base = reference(configs[ref]['cfg'], c1c2, df[KEY].drop_duplicates())
        lines += ['**Chạy lại ô tham chiếu so với C1/C2** (cùng block; chênh = chạy lại − C1/C2, theo từng arm)', '',
                  '| Arm | Chạy lại | C1/C2 | Chênh |', '|---|---:|---:|---:|']
        for a in configs[ref]['cfg']['arms']:
            d = describe(paired(arm(df, a), arm(base, a)))
            lines.append(f"| `{a}` | {arm(df, a)[Y].mean():.2f} | {arm(base, a)[Y].mean():.2f} | {fmt(d)} |")
        for comparison in ('dt_queue - edf', 'dt_queue - immediate', 'dt_queue - rows_legacy'):
            other = comparison.split(' - ')[1]
            lines.append(f"| {comparison} | {fmt(cell(ref, comparison))} | "
                         f"{fmt(describe(paired(arm(base, 'dt_queue'), arm(base, other))))} | |")
        lines.append('')

    # S2, S3, S5 against the C1/C2 blocks
    for stage, title in (('s2', 'S2: mức stress'), ('s3', 'S3: số yêu cầu K'), ('s5', 'S5: kích thước MLP')):
        names = [n for n, (s, _) in stages().items() if s == stage and n in configs]
        if not names:
            continue
        lines += [f'## {title}', '', 'Tham chiếu = các block tương ứng của C1/C2 (cùng seed, họ, tải, môi trường). '
                  + ('Với S3, cột "/yêu cầu" chia chênh lệch cho K để so được giữa các K.' if stage == 's3' else ''), '',
                  '| Config | Phạm vi | So sánh | Config | Tham chiếu C1/C2 |' + (' Config /yêu cầu | Tham chiếu /yêu cầu |' if stage == 's3' else ''),
                  '|---|---|---|---:|---:|' + ('---:|---:|' if stage == 's3' else '')]
        for name in names:
            df, cfg = configs[name]['df'], configs[name]['cfg']
            base = reference(cfg, c1c2, df[KEY].drop_duplicates())
            subsets = [('all', df, base)] + [(f'tải {l:g}', df[df.load == l], base[base.load == l]) for l in cfg['loads']]
            if len(cfg['environments']) > 1:
                subsets += [(f'{e}', df[df.environment == e], base[base.environment == e]) for e in cfg['environments']]
            for label, sub, bsub in subsets:
                for other in ('edf', 'immediate', 'rows_legacy'):
                    dc = describe(paired(arm(sub, 'dt_queue'), arm(sub, other)))
                    dr = describe(paired(arm(bsub, 'dt_queue'), arm(bsub, other)))
                    row = f'| `{name}` | {label} | dt_queue − {other} | {fmt(dc)} | {fmt(dr)} |'
                    if stage == 's3':
                        k = cfg['requests']
                        row += f" {dc['mean']/k:+.3f} | {dr['mean']/12:+.3f} |"
                    lines.append(row)
        lines.append('')
        if stage == 's5':
            lines += ['**Utility (macro-F1 calibration, trung bình theo arm)**', '', '| Config | Arm | Config | Tham chiếu C1/C2 |',
                      '|---|---|---:|---:|']
            for name in names:
                df, cfg = configs[name]['df'], configs[name]['cfg']
                base = reference(cfg, c1c2, df[KEY].drop_duplicates())
                for a in cfg['arms']:
                    lines.append(f"| `{name}` | `{a}` | {arm(df, a)['calibration_macro_f1'].mean():.4f} | "
                                 f"{arm(base, a)['calibration_macro_f1'].mean():.4f} |")
            lines.append('')

    # S4: within-arm change vs s4_ref on the development seeds
    if 's4_ref' in configs:
        ref_df = configs['s4_ref']['df']
        lines += ['## S4: tham số twin (seed phát triển 75042–75043, tải 0,2 F, 2 môi trường stress)', '',
                  'Chênh = giá trị mới − tham chiếu, cùng arm, ghép cặp theo block. "Ứng viên" theo quy tắc chốt trước: '
                  'khoảng tin cậy 95 % nằm hoàn toàn trên 0 cho arm dùng tham số đó.', '',
                  '| Config | Tham số | `dt_queue` | `edf` | `rows_legacy` | dt_queue − edf trong config | Ứng viên? |',
                  '|---|---|---:|---:|---:|---:|---|']
        lines.append(f"| `s4_ref` | tham chiếu | {arm(ref_df, 'dt_queue')[Y].mean():.2f} | {arm(ref_df, 'edf')[Y].mean():.2f} | "
                     f"{arm(ref_df, 'rows_legacy')[Y].mean():.2f} | {fmt(cell('s4_ref', 'dt_queue - edf'))} | |")
        for name, (stage, _) in stages().items():
            if stage != 's4' or name == 's4_ref' or name not in configs:
                continue
            df = configs[name]['df']
            short = next(k for k in TWIN_PARAMS if name[3:].startswith(k) and name[3+len(k)].isdigit())
            key, judged = TWIN_PARAMS[short]
            ds_ = {a: describe(paired(arm(df, a), arm(ref_df, a))) for a in ('dt_queue', 'edf', 'rows_legacy')}
            candidate = [a for a in judged if ds_[a]['n'] and ds_[a]['lo'] > 0]
            lines.append(f"| `{name}` | `{key}` = {configs[name]['cfg'][key]} | {fmt(ds_['dt_queue'])} | {fmt(ds_['edf'])} | "
                         f"{fmt(ds_['rows_legacy'])} | {fmt(cell(name, 'dt_queue - edf'))} | "
                         f"{'có (' + ', '.join(candidate) + ')' if candidate else 'không'} |")
        lines.append('')
    (analysis/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print(f'{done}/{total} configs complete; wrote {analysis}')


if __name__ == '__main__':
    main()
