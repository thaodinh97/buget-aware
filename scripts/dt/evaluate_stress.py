"""Offline evaluation of the Digital Twin's budget decisions on measured stress trajectories (directions 1 and 2 of
docs/dt_extensions.md). Usage:

  python scripts/dt/evaluate_stress.py runs/dt_stress results/dt_extensions/stress

ROOT holds one directory per (S, seed, family) written by scripts/dt/run_stress_trajectories.py. For every trajectory,
environment, budget level and planner the decisions are replayed on the measured costs (src/dt/planner.simulate): every
planner faces the same requests in the same states with the same true costs, so the comparison is paired.

Budget levels (src/dt/planner.BUDGET_LEVELS), from the measured fit of the same process, before any request: x .5 (tight),
.75 (mid), 1.5 (wide) the fit time of the largest shard's 20 rounds. Under no stress: wide fits every shard_full; mid forces
the largest shard to shard_fixed; tight sits at the fixed cost of the largest shard, so those decisions are knife-edge.

Scenarios of the headline tables: none, slow_clients, low_bandwidth (twin up to date, lag 0) and stale_state = the
transient_slow environment seen by a twin 3 requests behind (lag 3). Every (environment, lag) pair is in the CSV files.

Writes OUT/summary.csv (per trajectory), OUT/pooled.csv (counts pooled over the trajectories of a scenario),
OUT/forecast_errors.csv, OUT/report.md and, with --decisions, OUT/decisions.csv (one row per request and planner). The
'rescored' margin variant of Adaptive was added after the first evaluation (see docs/dt_extensions.md). Descriptive only:
no gate.
"""
import argparse
import json
import sys
from itertools import product
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import pandas as pd

from src.dt.planner import (BASELINE_PREDICTORS, BUDGET_LEVELS, FIXED_POLICIES, PlannerConfig, fit_ratios, rates, simulate,
                            summarize)

LAGS = (0, 3)
ALPHAS = (.6, 1.)
NO_TWIN = ('oracle',) + FIXED_POLICIES           # no twin state: lag and mode do not apply (one row, used in every scenario)
LABELS = {'oracle': 'Oracle', 'static': 'Static', 'adaptive': 'Adaptive', 'always_full': 'Luôn full',
          'always_fixed': 'Luôn fixed', 'rows_static': 'Theo dòng (tĩnh)', 'rows_ewma': 'Theo dòng (EWMA)',
          'last_shard': 'Lần trước cùng shard'}
SCENARIOS = {'none': ('none', 0), 'slow_clients': ('slow_clients', 0), 'low_bandwidth': ('low_bandwidth', 0),
             'stale_state': ('transient_slow', 3), 'transient_fresh': ('transient_slow', 0)}


def configs():
    out = [PlannerConfig('oracle')]
    for predictor, q, mode, lag in product(('static', 'adaptive'), (None, .9), ('precommit', 'midcourse'), LAGS):
        out.append(PlannerConfig(predictor, q, lag, mode))
        if predictor == 'adaptive' and q is not None:        # added after the first results; identical to 'ratio' for Static
            out.append(PlannerConfig(predictor, q, lag, mode, margin_mode='rescored'))
    # simple comparators (src/dt/planner.py), added 2026-11-16: no decision at all, or a simpler forecast in the same rules
    out += [PlannerConfig(policy) for policy in FIXED_POLICIES]
    for predictor, q, mode, lag in product(BASELINE_PREDICTORS, (None, .9), ('precommit', 'midcourse'), LAGS):
        out.append(PlannerConfig(predictor, q, lag, mode))
    # sensitivity probe, added after the comparators were seen: the EWMA weight of Adaptive (.3 everywhere above)
    for alpha, mode, lag in product(ALPHAS, ('precommit', 'midcourse'), LAGS):
        out.append(PlannerConfig('adaptive', None, lag, mode, alpha=alpha))
        out.append(PlannerConfig('adaptive', .9, lag, mode, margin_mode='rescored', alpha=alpha))
    return out


def load_run(path):
    run = json.loads((path/'run.json').read_text())
    fit_rounds = pd.read_csv(path/'fit_round_costs.csv').to_dict('records')
    fit_clients = pd.read_csv(path/'fit_client_costs.csv').to_dict('records')
    profile = rates(fit_rounds, fit_clients)
    envs = {e: [json.loads(line) for line in (path/e/'requests.jsonl').read_text().splitlines()]
            for e in run['envs'] if (path/e/'requests.jsonl').exists()}
    return run, profile, fit_ratios(fit_rounds, fit_clients, profile), envs, (fit_rounds, fit_clients)


def evaluate(root):
    decisions, summary = [], []
    for path in sorted(p for p in Path(root).iterdir() if (p/'run.json').exists()):
        run, profile, history, envs, fit_records = load_run(path)
        largest = run['fit']['largest_shard_full_seconds']
        for env, requests in envs.items():
            for level, scale in BUDGET_LEVELS.items():
                budget = scale*largest
                for cfg in configs():
                    rows = simulate(requests, budget, cfg, profile, history, fit_records)
                    label = {'run': path.name, 'seed': run['seed'], 'family': run['family'], 'env': env, 'lag': cfg.lag,
                             'budget_level': level, 'budget_seconds': budget, 'planner': cfg.predictor,
                             'margin': cfg.margin_q is not None, 'margin_mode': cfg.margin_mode if cfg.margin_q is not None else '',
                             'mode': 'precommit' if cfg.predictor in NO_TWIN else cfg.mode,
                             'alpha': cfg.alpha if cfg.predictor in ('adaptive', 'rows_ewma') else float('nan')}
                    decisions += [{**label, **r} for r in rows]
                    summary.append({**label, **summarize(rows)})
    return pd.DataFrame(decisions), pd.DataFrame(summary)


def pooled(decisions):
    rows = []
    keys = ['budget_level', 'planner', 'margin', 'margin_mode', 'mode', 'alpha']
    for name, (env, lag) in SCENARIOS.items():
        sel = decisions[(decisions.env == env) & ((decisions.lag == lag) | decisions.planner.isin(NO_TWIN))]
        for key, group in sel.groupby(keys, sort=False, dropna=False):
            rows.append({'scenario': name, **dict(zip(keys, key)), 'trajectories': group.run.nunique(),
                         **summarize(group.to_dict('records'))})
    return pd.DataFrame(rows)


def md(frame, index=True):
    """Markdown table without the optional tabulate dependency."""
    frame = frame.reset_index() if index else frame
    cols = [str(c) for c in frame.columns]
    cell = lambda v: f'{v:.3g}' if isinstance(v, float) else str(v)
    rows = ['| ' + ' | '.join(cols) + ' |', '|' + '---|'*len(cols)]
    rows += ['| ' + ' | '.join(cell(v) for v in r) + ' |' for r in frame.itertuples(index=False)]
    return chr(10).join(rows)


def planner_label(r):
    alpha = f" α={r['alpha']:g}" if r['planner'] == 'adaptive' and r['alpha'] != .3 else ''
    return f"{LABELS[r['planner']]}{alpha}{' + biên' if r['margin'] else ''}{' chấm lại' if r['margin_mode'] == 'rescored' else ''}"


MAIN = ['Oracle', 'Static', 'Static + biên', 'Adaptive', 'Adaptive + biên', 'Adaptive + biên chấm lại']
SIMPLE = ['Luôn full', 'Luôn fixed', 'Theo dòng (tĩnh)', 'Theo dòng (EWMA)', 'Lần trước cùng shard', 'Static', 'Adaptive']
SIMPLE_MARGIN = ['Theo dòng (EWMA) + biên', 'Lần trước cùng shard + biên', 'Adaptive + biên', 'Adaptive + biên chấm lại']
SENSITIVITY = ['Lần trước cùng shard', 'Adaptive', 'Adaptive α=0.6', 'Adaptive α=1', 'Adaptive + biên chấm lại',
               'Adaptive α=1 + biên chấm lại']


def table(frame, mode, value, order=MAIN):
    sel = frame[(frame['mode'] == mode) | frame.planner.isin(NO_TWIN)].copy()
    sel['planner_label'] = sel.apply(planner_label, axis=1)
    fmt = {'violation': lambda r: f"{r['violations']}/{r['accepted']}", 'false_reject': lambda r: f"{r['false_rejects']}/{r['oracle_accepted']}",
           'downgrade': lambda r: f"{r['downgrades']}/{r['oracle_full']}", 'overrun': lambda r: f"{r['overrun_seconds']:.0f}",
           'all': lambda r: f"{r['violations']}/{r['accepted']} · {r['false_rejects']} · {r['downgrades']}"}[value]
    sel['cell'] = sel.apply(fmt, axis=1)
    out = sel.pivot_table(index=['scenario', 'budget_level'], columns='planner_label', values='cell', aggfunc='first')
    out = out[[c for c in order if c in out.columns]]
    levels = list(BUDGET_LEVELS)
    out = out.reindex(sorted(out.index, key=lambda i: (list(SCENARIOS).index(i[0]), levels.index(i[1]))))
    return out


def forecast_errors(decisions):
    """Forecast of the shard_full option against its measured time, for the planners that observe every 20-round run
    (precommit, no margin, wide budget: every request is accepted or escalated as shard_full)."""
    sel = decisions[(decisions['mode'] == 'precommit') & ~decisions.margin & (decisions.budget_level == 'wide')
                    & decisions.planner.isin(['static', 'adaptive', *BASELINE_PREDICTORS])].copy()
    sel['abs_err'] = (sel.forecast_full - sel.true_full).abs()
    sel['rel_err'] = sel.true_full/sel.forecast_full - 1
    sel['planner'] = sel.planner + sel.alpha.map(lambda a: f'_a{a:g}' if a in ALPHAS else '')
    return sel.groupby(['env', 'lag', 'planner']).agg(requests=('step', 'size'), mae_seconds=('abs_err', 'mean'),
                                                      rel_err_mean=('rel_err', 'mean'), rel_err_max=('rel_err', 'max')).reset_index()


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--decisions', action='store_true', help='also write decisions.csv (one row per request and planner)')
    args = parser.parse_args(argv)
    decisions, summary = evaluate(args.root)
    if decisions.empty:
        raise SystemExit(f'no trajectories in {args.root}')
    args.output.mkdir(parents=True, exist_ok=True)
    if args.decisions:
        decisions.to_csv(args.output/'decisions.csv', index=False, float_format='%.6g')
    summary.to_csv(args.output/'summary.csv', index=False, float_format='%.6g')
    pool = pooled(decisions)
    pool.to_csv(args.output/'pooled.csv', index=False, float_format='%.6g')
    errors = forecast_errors(decisions)
    errors.to_csv(args.output/'forecast_errors.csv', index=False, float_format='%.6g')
    runs = sorted(summary.run.unique())
    lines = [f'# Stress, biên an toàn và điều chỉnh giữa chừng: bảng tổng hợp', '',
             f'Trajectory: {", ".join(runs)}. Mỗi ô gộp mọi trajectory của kịch bản; mỗi trajectory có 20 yêu cầu.', '']
    titles = {'violation': 'Vi phạm ngân sách (số yêu cầu vượt / số yêu cầu chấp nhận)',
              'false_reject': 'Từ chối nhầm (số yêu cầu bị từ chối dù oracle chấp nhận / số yêu cầu oracle chấp nhận)',
              'downgrade': 'Hạ cấp nhầm (chọn fixed dù full vừa ngân sách / số yêu cầu oracle chọn full)',
              'overrun': 'Tổng số giây vượt ngân sách'}
    for mode in ('precommit', 'midcourse'):
        lines += [f'## Chế độ `{mode}`', '']
        for value, title in titles.items():
            lines += [f'### {title}', '', md(table(pool, mode, value)), '']
    lines += ['## Đối chứng đơn giản (thêm 2026-11-16)', '',
              'Mỗi ô: vi phạm / chấp nhận · từ chối nhầm · hạ cấp nhầm. "Luôn full" và "Luôn fixed" không dự báo và không '
              'từ chối; ba cột "Theo dòng" và "Lần trước cùng shard" chỉ thay phần dự báo, giữ nguyên quy tắc quyết định.', '']
    for mode in ('precommit', 'midcourse'):
        lines += [f'### `{mode}`, không biên', '', md(table(pool, mode, 'all', SIMPLE)), '',
                  f'### `{mode}`, có biên q90', '', md(table(pool, mode, 'all', SIMPLE_MARGIN)), '']
    lines += ['## Độ nhạy theo trọng số EWMA của Adaptive (thăm dò, thêm sau khi xem bảng đối chứng)', '',
              'Mỗi ô: vi phạm / chấp nhận · từ chối nhầm · hạ cấp nhầm. α = 0,3 là giá trị của `shard_dt.py`; α = 1 giữ '
              'giá trị quan sát gần nhất của từng client và của phần server.', '']
    for mode in ('precommit', 'midcourse'):
        lines += [f'### `{mode}`', '', md(table(pool, mode, 'all', SENSITIVITY)), '']
    lines += ['## Sai số dự báo của shard_full (precommit, không biên, ngân sách rộng)', '', md(errors, index=False), '']
    (args.output/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    print(f'wrote {(args.output/"report.md").as_posix()}: {len(summary)} (trajectory, environment, budget, planner) rows')


if __name__ == '__main__':
    main()
