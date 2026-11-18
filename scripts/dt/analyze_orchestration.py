"""Audit and the predeclared tests of the C1/C2 orchestration protocol (docs/dt_orchestration_plan.md).

    python scripts/dt/analyze_orchestration.py results/dt_orchestration

Reads only the run directory (no training). Unit of analysis: the block (seed x family x load x environment); every
arm in a block serves the same requests from the same fitted ensemble. Primary metric: joint-success requests per
block (on time and within the pooled round budget).
- C1, per baseline b: d = proposed - b; one-sided Wilcoxon signed-rank (H1: d > 0), Holm over the baselines.
  b is beaten when the Holm-adjusted p < alpha and mean d > 0. C1 holds when every baseline is beaten, the mean paired
  calibration macro-F1 difference to each baseline is >= the predeclared floor, and every chain is exact.
- C2, per ablation a: d = proposed - a; two-sided Wilcoxon, Holm over the ablations; a component 'contributes' when
  Holm p < alpha and mean d > 0, is 'harmful' when Holm p < alpha and mean d < 0, otherwise 'not detected'.
  95 % percentile bootstrap intervals over blocks are reported for every mean difference.
Writes report.md, paired.csv, tests.json into the run directory.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
from scipy import stats

from scripts.dt.run_orchestration import source_hashes


def holm(pvalues):
    """Holm step-down adjusted p-values, same order as the input."""
    p = np.asarray(pvalues, float)
    order = np.argsort(p)
    adjusted = np.empty_like(p)
    running = 0.
    for rank, i in enumerate(order):
        running = max(running, min(1., (len(p)-rank)*p[i]))
        adjusted[i] = running
    return adjusted.tolist()


def wilcoxon(d, alternative):
    d = np.asarray(d, float)
    if not np.any(d):
        return 1.
    return float(stats.wilcoxon(d, zero_method='wilcox', alternative=alternative).pvalue)


def bootstrap_ci(d, resamples, seed):
    d = np.asarray(d, float)
    rng = np.random.default_rng(seed)
    means = d[rng.integers(0, len(d), size=(resamples, len(d)))].mean(axis=1)
    return [float(np.quantile(means, .025)), float(np.quantile(means, .975))]


def audit_arm(directory, requests):
    """Recompute the arm from its event log: every request once, no future arrival, one worker, latency = wait +
    service, joint success as summarized. Returns the recomputed joint-success count."""
    events = [json.loads(line) for line in (directory/'events.jsonl').read_text(encoding='utf-8').splitlines()]
    served = sorted(s for e in events for s in e['steps'])
    if served != sorted(r['step'] for r in requests):
        raise ValueError(f'{directory}: requests not served exactly once')
    arrival = {r['step']: r['time'] for r in requests}
    previous_ready = -np.inf
    joint = 0
    for e in events:
        if any(arrival[s] > e['started_seconds']+1e-9 for s in e['steps']):
            raise ValueError(f'{directory}: a batch contains a request that had not arrived')
        if e['started_seconds'] < previous_ready-1e-9:
            raise ValueError(f'{directory}: overlapping executions on one worker')
        previous_ready = e['model_ready_seconds']
        if not np.allclose(e['latency_seconds'], [w+e['service_seconds'] for w in e['waiting_seconds']]):
            raise ValueError(f'{directory}: latency != waiting + service')
        if not (e['exact'] and e['structure_ok']):
            raise ValueError(f'{directory}: an operation was not exact')
        if not e['round_over_budget']:
            joint += sum(t == 0 for t in e['lateness_seconds'])
    final = json.loads((directory/'final.json').read_text(encoding='utf-8'))['summary']
    if final['joint_success_requests'] != joint or not final['bit_exact']:
        raise ValueError(f'{directory}: summary disagrees with events or final model not bit-exact')
    return joint


def utility_floors(cfg):
    """Per-baseline floor of the mean paired calibration macro-F1 difference (proposed - baseline).

    'calibration_f1_floor' maps each C1 baseline to its floor (protocol, after the smoke). The smoke config was frozen
    before that change and has one 'min_mean_paired_calibration_f1_delta' for every baseline; it is read as such."""
    a = cfg['analysis']
    if 'calibration_f1_floor' in a:
        floors = a['calibration_f1_floor']
    else:
        floors = dict.fromkeys(cfg['c1_baselines'], a['min_mean_paired_calibration_f1_delta'])
    if set(floors) != set(cfg['c1_baselines']):
        raise ValueError('a utility floor is needed for every C1 baseline, and only for them')
    return floors


def paired(frame, proposed, others, metric):
    wide = frame.pivot(index='block', columns='predictor', values=metric)
    return {o: (wide[proposed]-wide[o]).to_numpy() for o in others}


def analyze(root, smoke=False):
    lock = json.loads((root/'protocol_lock.json').read_text(encoding='utf-8'))
    cfg = lock['config']
    if smoke and not cfg['purpose'].startswith('SMOKE ONLY'):
        raise ValueError('--smoke is only for runs of the smoke config')
    config_path = next(ROOT/p for p in lock['source_sha256_lf'] if p.startswith('configs/'))
    plan_path = next(ROOT/p for p in lock['source_sha256_lf'] if p.startswith('docs/'))
    current = source_hashes(config_path, plan_path)
    changed = sorted(p for p, h in lock['source_sha256_lf'].items() if current.get(p) != h)
    if changed and not smoke:
        raise ValueError(f'code/config changed after the freeze: {changed}')
    rows = []
    for block in sorted(p for p in root.iterdir() if (p/'complete.json').exists()):
        run = json.loads((block/'run.json').read_text(encoding='utf-8'))
        summaries = json.loads((block/'complete.json').read_text(encoding='utf-8'))['summaries']
        if sorted(s['predictor'] for s in summaries) != sorted(cfg['arms']):
            raise ValueError(f'{block}: arms differ from the config')
        for s in summaries:
            audit_arm(block/s['predictor'], run['requests'])
        rows.extend(summaries)
    frame = pd.DataFrame(rows)
    expected = len(cfg['seeds'])*len(cfg['families'])*len(cfg['loads'])*len(cfg['environments'])
    blocks = frame['block'].nunique()
    a = cfg['analysis']
    proposed = cfg['proposed']
    tests = dict(blocks=blocks, expected_blocks=expected, complete=blocks == expected,
                 all_exact=bool(frame['bit_exact'].all() and frame['exact_all_events'].all()),
                 smoke=smoke, sources_changed_since_lock=changed)
    others = cfg['c1_baselines']+cfg['c2_ablations']
    joint = paired(frame, proposed, others, 'joint_success_requests')
    f1 = paired(frame, proposed, others, 'calibration_macro_f1')
    index = frame.pivot(index='block', columns='predictor', values='joint_success_requests').index
    load_of = frame.drop_duplicates('block').set_index('block')['load'].reindex(index).to_numpy()

    def describe(name, d, p, p_holm):
        per_load = {f'{load:g}': (float(d[load_of == load].mean()) if (load_of == load).any() else None)
                    for load in cfg['loads']}
        return dict(mean_delta=float(np.mean(d)), ci95=bootstrap_ci(d, a['bootstrap_resamples'], a['bootstrap_seed']),
                    wins=int((d > 0).sum()), ties=int((d == 0).sum()), losses=int((d < 0).sum()),
                    p=p, p_holm=p_holm, mean_delta_by_load=per_load,
                    mean_calibration_f1_delta=float(np.mean(f1[name])))

    floors = utility_floors(cfg)
    c1_p = [wilcoxon(joint[b], 'greater') for b in cfg['c1_baselines']]
    c1 = {}
    for b, p, ph in zip(cfg['c1_baselines'], c1_p, holm(c1_p)):
        row = describe(b, joint[b], p, ph)
        row['beaten'] = bool(ph < a['alpha'] and row['mean_delta'] > 0)
        row['calibration_f1_floor'] = floors[b]
        row['utility_ok'] = bool(row['mean_calibration_f1_delta'] >= floors[b])
        c1[b] = row
    c2_p = [wilcoxon(joint[x], 'two-sided') for x in cfg['c2_ablations']]
    c2 = {}
    for x, p, ph in zip(cfg['c2_ablations'], c2_p, holm(c2_p)):
        row = describe(x, joint[x], p, ph)
        row['verdict'] = ('contributes' if ph < a['alpha'] and row['mean_delta'] > 0 else
                          'harmful' if ph < a['alpha'] and row['mean_delta'] < 0 else 'not detected')
        c2[x] = row
    tests.update(c1=c1, c2=c2, c1_holds=bool(tests['complete'] and tests['all_exact'] and
                                             all(r['beaten'] and r['utility_ok'] for r in c1.values())))
    per_arm = frame.groupby('predictor').agg(
        requests=('requests', 'sum'), joint_success=('joint_success_requests', 'sum'),
        deadline_misses=('deadline_misses', 'sum'), service_seconds=('service_seconds', 'sum'),
        latency_p95_mean=('latency_p95_seconds', 'mean'), full_requests=('full_requests', 'sum'),
        calibration_f1=('calibration_macro_f1', 'mean'), calibration_fpr=('calibration_fpr', 'mean'),
        threshold_seconds=('threshold_seconds_total', 'sum'), plan_seconds=('plan_seconds_total', 'sum'))
    per_arm['joint_success_rate'] = per_arm['joint_success']/per_arm['requests']
    pairs = pd.DataFrame([dict(block=k, load=load, comparator=o, joint_delta=v, calibration_f1_delta=f)
                          for o in others for k, load, v, f in zip(index, load_of, joint[o], f1[o])])
    pairs.to_csv(root/'paired.csv', index=False)
    (root/'tests.json').write_text(json.dumps(tests, indent=2), encoding='utf-8')
    (root/'report.md').write_text(markdown(cfg, tests, per_arm), encoding='utf-8')
    return tests, per_arm


def markdown(cfg, tests, per_arm):
    lines = ['# C1/C2 orchestration: generated report', '']
    if tests['smoke']:
        lines += ['**SMOKE RUN: plumbing and timing only, never evidence.** '
                  f"Sources changed since its lock: {len(tests['sources_changed_since_lock'])}.", '']
    lines += [f"Blocks: {tests['blocks']} / {tests['expected_blocks']}; all chains exact: {tests['all_exact']}; "
             f"**C1 holds: {tests['c1_holds']}**. Tests and thresholds are those of the protocol "
             '(docs/dt_orchestration_plan.md); this file is generated.', '',
             '## Per arm', '',
             '| Arm | Joint success | Rate | Deadline misses | Service s | Mean p95 latency s | Full | Cal. F1 | Cal. FPR |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for name, r in per_arm.sort_values('joint_success_rate', ascending=False).iterrows():
        lines.append(f"| `{name}` | {int(r.joint_success)}/{int(r.requests)} | {r.joint_success_rate:.3f} | "
                     f"{int(r.deadline_misses)} | {r.service_seconds:.1f} | {r.latency_p95_mean:.2f} | "
                     f"{int(r.full_requests)} | {r.calibration_f1:.4f} | {r.calibration_fpr:.4f} |")
    for title, key, verdict in (('C1: proposed vs baselines (one-sided, Holm)', 'c1', 'beaten'),
                                ('C2: proposed vs ablations (two-sided, Holm)', 'c2', 'verdict')):
        lines += ['', f'## {title}', '',
                  '| Comparator | Mean Δ joint/block | 95 % CI | W/T/L | p | p Holm | Δ cal. F1 (floor) | Δ by load | Verdict |',
                  '|---|---:|---|---|---:|---:|---:|---|---|']
        for name, r in tests[key].items():
            loads = ', '.join(f'{k}: {v:+.2f}' for k, v in r['mean_delta_by_load'].items() if v is not None)
            floor = f" ({r['calibration_f1_floor']:+.3f}, {'ok' if r['utility_ok'] else 'FAIL'})" if key == 'c1' else ''
            lines.append(f"| `{name}` | {r['mean_delta']:+.2f} | [{r['ci95'][0]:+.2f}, {r['ci95'][1]:+.2f}] | "
                         f"{r['wins']}/{r['ties']}/{r['losses']} | {r['p']:.4f} | {r['p_holm']:.4f} | "
                         f"{r['mean_calibration_f1_delta']:+.4f}{floor} | {loads} | {r[verdict]} |")
    return '\n'.join(lines)+'\n'


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('root', type=Path)
    parser.add_argument('--smoke', action='store_true',
                        help='smoke runs only: report source changes since the lock instead of refusing')
    args = parser.parse_args(argv)
    tests, per_arm = analyze(args.root, smoke=args.smoke)
    print(per_arm.to_string())
    print(json.dumps({k: tests[k] for k in ('blocks', 'expected_blocks', 'all_exact', 'c1_holds')}, indent=2))


if __name__ == '__main__':
    main()
