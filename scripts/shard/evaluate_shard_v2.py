"""Gates of the v2 shard protocol (docs/shard_v2_plan.md), fixed before the v2 runs. Usage:
python scripts/shard/evaluate_shard_v2.py ENSEMBLE_DIR STREAM_DIR [OUT_JSON]
(ENSEMBLE_DIR from run_shard_ensemble_check_v2.py, STREAM_DIR from run_shard_stream_v2.py.)

Metrics on the calibration split. Comparisons are paired by training seed and use a one-sided noninferiority test
(shard_v2.noninferior, paired t, alpha 0.05): macro-F1 passes iff mean(d) - t_{0.95,n-1} se(d) > -D_F1, FPR passes iff
mean(d) + t_{0.95,n-1} se(d) < D_FPR, d = candidate - reference. A comparison passes iff both pass. A gate passes iff all of
its comparisons and checks pass (intersection-union, so no multiplicity adjustment within a gate).
G1  ensemble utility (run 1)       S in {2,4,5} vs S = 1; no (S, seed) with FPR > HIGH_FPR
G2  one exact unlearning (run 1)   at R_base, S in {2,4,5} x {client, random50} vs S = 1 with the same request; no row at
                                   R_base with FPR > HIGH_FPR
G3  exactness (run 2)              every policy except none (utility_stop_v1 included): exact and structure_ok at every step;
                                   bit-exact, included = active shards and structure_ok at the end; complete coverage
G4  healthy reference (run 2)      full_retrain_single has FPR <= HIGH_FPR at every step and seed
G5a vs exact shard retrain         S in {4,5}, shard_fixed / dynamicfu / utility_stop / drop_backfill (after its backlog), final
                                   state vs shard_full's final state (same S, same seed); no seed with final FPR > HIGH_FPR
G5b vs single-model retrain        S in {4,5}, shard_full / shard_fixed / dynamicfu / utility_stop / drop_backfill, final state
                                   vs full_retrain_single at step 10; no seed with final FPR > HIGH_FPR
G6  utility_stop safety            utility_stop (v2) has FPR <= HIGH_FPR at every step 1-10, every S and seed
Integrity: both runs used design 'v2' (config 'standard', seeds 50042..50051), the expected data content hash, and the SHA-256
of every tracked file recorded at the start equals the current file. Paths are compared after normalisation (shard_v2.norm_rel).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.shard.shard_v2 import ROOT, TRACKED, DATA_SHA256, DATA_DEFAULT, DESIGNS, text_sha256, norm_rel, noninferior

HIGH_FPR, D_F1, D_FPR, ALPHA = .45, .01, .02, .05
SHARDS_1, REQUESTS, SHARDS_2, STEPS = [2, 4, 5], ['client', 'random50'], [4, 5], 10
EXACT = ['shard_full', 'shard_fixed', 'dynamicfu', 'utility_stop', 'drop_backfill']
REDUCED = ['shard_fixed', 'dynamicfu', 'utility_stop', 'drop_backfill']
CHECKED = EXACT + ['utility_stop_v1']
RUNNERS = {'ensemble': 'scripts/shard/run_shard_ensemble_check_v2.py', 'stream': 'scripts/shard/run_shard_stream_v2.py'}
DESIGN = DESIGNS['v2']


def compare(x, ref, seeds, label):
    if sorted(x.index) != sorted(seeds) or sorted(ref.index) != sorted(seeds):
        raise ValueError(f'{label}: seeds {sorted(x.index)} / {sorted(ref.index)} != {sorted(seeds)}')
    ref = ref.loc[x.index]
    f1 = noninferior(x.cal_macro_f1 - ref.cal_macro_f1, D_F1, 'lower', ALPHA)
    fpr = noninferior(x.cal_fpr - ref.cal_fpr, D_FPR, 'higher', ALPHA)
    return {'dF1_mean': f1['mean'], 'dF1_lower95': f1['bound'], 'dFPR_mean': fpr['mean'], 'dFPR_upper95': fpr['bound'],
            'F1_mean': x.cal_macro_f1.mean(), 'FPR_max': x.cal_fpr.max(), 'ok': f1['ok'] and fpr['ok']}


def high(df, cols): return df[df.cal_fpr > HIGH_FPR][cols].to_dict('records')


def final(s, S, p):
    return s[(s.S == S) & (s.policy == p) & (s.step == (STEPS + 1 if p == 'drop_backfill' else STEPS))].set_index('seed')


def g1(e, seeds):
    ref = e[e.S == 1].set_index('seed')
    comps = [{'S': S, **compare(e[e.S == S].set_index('seed'), ref, seeds, f'G1 S={S}')} for S in SHARDS_1]
    hi = high(e, ['S', 'seed', 'cal_macro_f1', 'cal_fpr'])
    return {'pass': all(c['ok'] for c in comps) and not hi, 'comparisons': comps, 'high_fpr': hi}


def g2(u, seeds, rb):
    u = u[u.rounds == rb]; comps = []
    for req in REQUESTS:
        ref = u[(u.S == 1) & (u.request == req)].set_index('seed')
        comps += [{'request': req, 'S': S, **compare(u[(u.S == S) & (u.request == req)].set_index('seed'), ref, seeds, f'G2 {req} S={S}')}
                  for S in SHARDS_1]
    hi = high(u, ['S', 'seed', 'request', 'cal_macro_f1', 'cal_fpr'])
    return {'pass': all(c['ok'] for c in comps) and not hi, 'R_base': int(rb), 'comparisons': comps, 'high_fpr': hi}


def g3(s, x, seeds):
    want = {(S, p) for S in SHARDS_2 for p in CHECKED}
    st = s[s.policy.isin(CHECKED) & (s.step <= STEPS)]; xs = x[x.policy.isin(CHECKED)]
    per_step = st.groupby(['S', 'policy'])[['exact', 'structure_ok']].all()
    end = xs.groupby(['S', 'policy'])[['bit_exact', 'included_equals_active', 'structure_ok_end']].all()
    covered = (set(per_step.index) == want and set(end.index) == want and len(xs) == len(want)*len(seeds)
               and len(st) == len(want)*len(seeds)*STEPS)
    backlog = s[(s.policy == 'drop_backfill') & (s.step == STEPS + 1)]
    backlog_ok = len(backlog) == len(SHARDS_2)*len(seeds) and bool(backlog.exact.all() and backlog.structure_ok.all())
    bad = {c: [list(k) for k, v in per_step[c].items() if not v] for c in per_step} | {c: [list(k) for k, v in end[c].items() if not v] for c in end}
    return {'pass': bool(covered and backlog_ok and per_step.all().all() and end.all().all()), 'coverage_complete': bool(covered),
            'backlog_rows_ok': backlog_ok, 'failures': {k: v for k, v in bad.items() if v}}


def g4(s, seeds):
    ref = s[s.policy == 'full_retrain_single']
    if len(ref) != STEPS*len(seeds): raise ValueError(f'G4: {len(ref)} reference rows, expected {STEPS*len(seeds)}')
    hi = high(ref, ['seed', 'step', 'cal_macro_f1', 'cal_fpr'])
    return {'pass': not hi, 'F1_min': ref.cal_macro_f1.min(), 'FPR_max': ref.cal_fpr.max(), 'high_fpr': hi}


def g5(s, seeds, against):
    comps = []
    for S in SHARDS_2:
        if against == 'shard_full': ref, pols = final(s, S, 'shard_full'), REDUCED
        else: ref, pols = s[(s.policy == 'full_retrain_single') & (s.step == STEPS)].set_index('seed'), EXACT
        for p in pols:
            fin = final(s, S, p); c = compare(fin, ref, seeds, f'G5 {against} S={S} {p}')
            c['high_fpr_seeds'] = int((fin.cal_fpr > HIGH_FPR).sum()); c['ok'] = c['ok'] and c['high_fpr_seeds'] == 0
            comps.append({'S': S, 'policy': p, **c})
    return {'pass': all(c['ok'] for c in comps), 'against': against, 'comparisons': comps}


def g6(s):
    u = s[(s.policy == 'utility_stop') & (s.step <= STEPS)]
    hi = high(u, ['S', 'seed', 'step', 'cal_macro_f1', 'cal_fpr'])
    return {'pass': bool(len(u) > 0 and not hi), 'FPR_max': u.cal_fpr.max(), 'F1_min': u.cal_macro_f1.min(), 'high_fpr': hi}


def integrity(dirs):
    out = {}
    for name, d in dirs.items():
        p = d/'run.json'
        if not p.exists(): out[name] = {'ok': False, 'reason': 'no run.json'}; continue
        rec = json.loads(p.read_text()); sha = rec.get('sha256_at_start', {})
        mism = sorted(set(TRACKED) ^ set(sha)) + [f for f in TRACKED if f in sha and (not (ROOT/f).exists() or text_sha256(ROOT/f) != sha[f])]
        checks = {'design_v2': rec.get('design_name') == 'v2' and rec.get('design') == DESIGN,
                  'config_standard': rec.get('train_config_name') == 'standard',
                  'runner': norm_rel(ROOT/rec.get('runner', '')) == RUNNERS[name],
                  'data_content': rec.get('data_content_sha256') == DATA_SHA256,
                  'tracked_sha256': not mism}
        out[name] = {'ok': all(checks.values()), 'checks': checks, 'sha256_mismatch': mism,
                     'data_dir': norm_rel(rec['data_dir_abs']) if 'data_dir_abs' in rec else None,
                     'data_dir_is_default': ('data_dir_abs' in rec and norm_rel(rec['data_dir_abs']) == DATA_DEFAULT),
                     'data_content_sha256': rec.get('data_content_sha256')}
    return out


def descriptive(s):
    st = s[s.step <= STEPS]
    v1 = st[st.policy == 'utility_stop_v1'].groupby('S').cal_fpr.apply(lambda v: int((v > HIGH_FPR).sum())).to_dict()
    us = st[st.policy == 'utility_stop']
    fb = us.groupby('S').agg(fallback_rounds=('fallback_rounds', 'sum'), steps_with_fallback=('fallback_rounds', lambda v: int((v > 0).sum())),
                             condition_unmet=('condition_met', lambda v: int((~v.astype(bool)).sum()))).to_dict('index')
    ev = st[(st.policy == 'shard_full')].drop_duplicates(['S', 'step'])[['S', 'step', 'event']]
    return {'utility_stop_v1_steps_fpr_gt_high': v1, 'utility_stop_fallback': fb,
            'structure_events': ev[ev.event != 'retrain'].to_dict('records')}


def main():
    E, ST = Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve()
    e, u = pd.read_csv(E/'ensemble.csv'), pd.read_csv(E/'unlearning.csv')
    s, x = pd.read_csv(ST/'steps.csv'), pd.read_csv(ST/'exactness.csv')
    seeds = sorted(int(v) for v in e.seed.unique())
    for name, df in [('unlearning', u), ('steps', s), ('exactness', x)]:
        if sorted(int(v) for v in df.seed.unique()) != seeds: raise ValueError(f'{name}.csv seeds differ from ensemble.csv')
    rb = json.loads((E/'run.json').read_text())['train_config']['fl']['rounds'] if (E/'run.json').exists() else int(u.rounds.max())
    gates = {'G1': g1(e, seeds), 'G2': g2(u, seeds, rb), 'G3': g3(s, x, seeds), 'G4': g4(s, seeds),
             'G5a': g5(s, seeds, 'shard_full'), 'G5b': g5(s, seeds, 'full_retrain_single'), 'G6': g6(s)}
    integ = integrity({'ensemble': E, 'stream': ST}); integ_ok = all(v['ok'] for v in integ.values()) and seeds == DESIGN['seeds']
    us_rows = lambda g: all(c['ok'] for c in gates[g]['comparisons'] if c['policy'] == 'utility_stop')
    claims = {'sharding_keeps_single_model_utility (G1, G2, G5b)': gates['G1']['pass'] and gates['G2']['pass'] and gates['G5b']['pass'],
              'reduced_budget_policies_match_shard_full (G5a)': gates['G5a']['pass'],
              'exact_unlearning (G3)': gates['G3']['pass'],
              'utility_stop_v2_safe (G6 + its G5a/G5b rows)': gates['G6']['pass'] and us_rows('G5a') and us_rows('G5b')}
    pre = gates['G4']['pass'] and integ_ok
    result = {'confirmed_all': pre and all(g['pass'] for g in gates.values()), 'preconditions_ok (G4 + integrity)': pre,
              'claims': {k: v and pre for k, v in claims.items()}, 'seeds': seeds,
              'thresholds': {'HIGH_FPR': HIGH_FPR, 'D_F1': D_F1, 'D_FPR': D_FPR, 'alpha_one_sided': ALPHA},
              'gates': gates, 'integrity': integ, 'descriptive': descriptive(s),
              'inputs': {'ensemble': norm_rel(E), 'stream': norm_rel(ST)}}

    print(f'inputs {result["inputs"]}; seeds {seeds}; R_base {rb}')
    print(f'noninferiority: one-sided {int(100*(1-ALPHA))}% paired t; dF1 lower bound > -{D_F1}, dFPR upper bound < +{D_FPR}; FPR > {HIGH_FPR} = high-FPR mode')
    for k, g in gates.items():
        print(f'\n{k}: {"PASS" if g["pass"] else "FAIL"}')
        if 'comparisons' in g: print(pd.DataFrame(g['comparisons']).round(4).to_string(index=False))
        for key in ('high_fpr', 'failures'):
            if g.get(key): print(f'  {key}: {g[key] if isinstance(g[key], dict) else g[key][:10]}')
        if k == 'G3': print(f'  coverage complete: {g["coverage_complete"]}; backlog rows ok: {g["backlog_rows_ok"]}')
        if k in ('G4', 'G6'): print(f'  F1 min {g["F1_min"]:.4f}, FPR max {g["FPR_max"]:.4f}')
    print('\nintegrity:', {n: (v['ok'], v.get('checks', v.get('reason')), v.get('data_dir')) for n, v in integ.items()})
    print('descriptive:', json.dumps(result['descriptive'], default=str))
    print('\nclaims:'); [print(f'  {k}: {v}') for k, v in result['claims'].items()]
    print(f'\nCONFIRMED (all gates): {result["confirmed_all"]}  (preconditions G4 + integrity: {pre})')
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(result, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


if __name__ == '__main__':
    main()
