"""Gates of docs/shard_standard_config_plan.md, fixed before the confirmation runs. Usage:
python scripts/shard/evaluate_standard_config.py ENSEMBLE_DIR STREAM_DIR [OUT_JSON]
(ENSEMBLE_DIR from run_shard_ensemble_check.py, STREAM_DIR from run_shard_stream.py.)

Metrics are on the calibration split; differences are paired by training seed and the conditions use the mean over seeds.
G1 ensemble utility (run 1)      for every S in {2,4,5,10}: mean(F1_S - F1_1) >= -D_F1 and mean(FPR_S - FPR_1) <= D_FPR;
                                 no (S, seed), S = 1 included, with FPR > HIGH_FPR
G2 one exact unlearning (run 1)  at R = R_base, for every S in {2,4,5,10} and request in {client, random50}: the same against
                                 S = 1 with the same request (= full retrain on D minus F); no row at R_base with FPR > HIGH_FPR
G3 exactness (run 2)             every policy except none: exact at every step and bit-exact at the end, every S and seed
G4 healthy reference (run 2)     full_retrain_single has FPR <= HIGH_FPR at every step and seed
G5 final utility (run 2)         for S in {4,10} and every exact policy, the final state (drop_backfill: after its backlog) against
                                 full_retrain_single at step 10: mean(F1 - F1_ref) >= -D_F1 and mean(FPR - FPR_ref) <= D_FPR;
                                 no seed with final FPR > HIGH_FPR
Confirmed iff G1-G5 all pass on all seeds and the SHA-256 recorded at the start of both runs match the current files.
Sensitivity (pre-specified): G1, G2, G4, G5 again on the seeds other than 10042 (seed 10042 with S in {1, 4} under this
config was seen in diagnose_training_budget.py).
"""
import json, sys
from pathlib import Path
import numpy as np, pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from scripts.shard.train_config import ROOT, TRACKED, text_sha256

HIGH_FPR, D_F1, D_FPR = .45, .01, .02
SHARDS_1, REQUESTS, SHARDS_2, STEPS = [2, 4, 5, 10], ['client', 'random50'], [4, 10], 10
EXACT_POLICIES = ['shard_full', 'shard_fixed', 'dynamicfu', 'utility_stop', 'drop_backfill']
SEEN_SEED, CONFIG, DATA_NAME = 10042, 'standard', 'connected_v3_auxfit_seed42'


def compare(x, ref, seeds, label):
    """x, ref: frames indexed by seed with cal_macro_f1 / cal_fpr. Paired differences over exactly `seeds`."""
    if sorted(x.index) != sorted(seeds) or sorted(ref.index) != sorted(seeds):
        raise ValueError(f'{label}: seeds {sorted(x.index)} / {sorted(ref.index)} != {sorted(seeds)}')
    df1 = x.cal_macro_f1 - ref.loc[x.index].cal_macro_f1; dfpr = x.cal_fpr - ref.loc[x.index].cal_fpr
    return {'dF1_mean': df1.mean(), 'dF1_min': df1.min(), 'dFPR_mean': dfpr.mean(), 'dFPR_max': dfpr.max(),
            'F1_mean': x.cal_macro_f1.mean(), 'FPR_mean': x.cal_fpr.mean(), 'FPR_max': x.cal_fpr.max(),
            'ok': bool(df1.mean() >= -D_F1 and dfpr.mean() <= D_FPR)}


def high(df, cols): return df[df.cal_fpr > HIGH_FPR][cols].to_dict('records')


def g1(e, seeds):
    e = e[e.seed.isin(seeds)]; ref = e[e.S == 1].set_index('seed')
    comps = [{'S': S, **compare(e[e.S == S].set_index('seed'), ref, seeds, f'G1 S={S}')} for S in SHARDS_1]
    hi = high(e, ['S', 'seed', 'cal_macro_f1', 'cal_fpr'])
    return {'pass': all(c['ok'] for c in comps) and not hi, 'comparisons': comps, 'high_fpr': hi}


def g2(u, seeds, rb):
    u = u[u.seed.isin(seeds) & (u.rounds == rb)]; comps = []
    for req in REQUESTS:
        ref = u[(u.S == 1) & (u.request == req)].set_index('seed')
        comps += [{'request': req, 'S': S, **compare(u[(u.S == S) & (u.request == req)].set_index('seed'), ref, seeds, f'G2 {req} S={S}')}
                  for S in SHARDS_1]
    hi = high(u, ['S', 'seed', 'request', 'cal_macro_f1', 'cal_fpr'])
    return {'pass': all(c['ok'] for c in comps) and not hi, 'R_base': int(rb), 'comparisons': comps, 'high_fpr': hi}


def g3(s, x, seeds):
    want = {(S, p) for S in SHARDS_2 for p in EXACT_POLICIES}
    st = s[s.policy.isin(EXACT_POLICIES)]; step_ok = st.groupby(['S', 'policy']).exact.all()
    bit_ok = x.groupby(['S', 'policy']).bit_exact.all()
    covered = (set(step_ok.index) == want and set(bit_ok.index) == want and len(x) == len(want)*len(seeds)
               and len(st[st.step <= STEPS]) == len(want)*len(seeds)*STEPS)
    return {'pass': bool(covered and step_ok.all() and bit_ok.all()), 'coverage_complete': bool(covered),
            'not_exact_at_some_step': [list(k) for k, v in step_ok.items() if not v],
            'not_bit_exact': [list(k) for k, v in bit_ok.items() if not v]}


def g4(s, seeds):
    ref = s[(s.policy == 'full_retrain_single') & s.seed.isin(seeds)]
    if len(ref) != STEPS*len(seeds): raise ValueError(f'G4: {len(ref)} reference rows, expected {STEPS*len(seeds)}')
    hi = high(ref, ['seed', 'step', 'cal_macro_f1', 'cal_fpr'])
    return {'pass': not hi, 'FPR_max': ref.cal_fpr.max(), 'F1_min': ref.cal_macro_f1.min(), 'high_fpr': hi}


def g5(s, seeds):
    s = s[s.seed.isin(seeds)]; ref = s[(s.policy == 'full_retrain_single') & (s.step == STEPS)].set_index('seed'); comps = []
    for S in SHARDS_2:
        for p in EXACT_POLICIES:
            fin = s[(s.S == S) & (s.policy == p) & (s.step == (STEPS + 1 if p == 'drop_backfill' else STEPS))].set_index('seed')
            c = compare(fin, ref, seeds, f'G5 S={S} {p}'); c['ok'] = c['ok'] and bool((fin.cal_fpr <= HIGH_FPR).all())
            comps.append({'S': S, 'policy': p, **c})
    return {'pass': all(c['ok'] for c in comps), 'reference_F1': ref.cal_macro_f1.mean(), 'reference_FPR': ref.cal_fpr.mean(),
            'comparisons': comps}


def integrity(dirs):
    out = {}
    for name, d in dirs.items():
        cfg_p, data_p = Path(d)/'train_config.json', Path(d)/'data.json'
        if not cfg_p.exists(): out[name] = {'ok': False, 'reason': 'no train_config.json'}; continue
        cfg = json.loads(cfg_p.read_text()); data = json.loads(data_p.read_text()) if data_p.exists() else {}
        sha = cfg.get('sha256_at_start', {})
        mism = [f for f in TRACKED if sha.get(f) != (text_sha256(ROOT/f) if (ROOT/f).exists() else None)]
        mism += [f for f in sha if f not in TRACKED and (not (ROOT/f).exists() or text_sha256(ROOT/f) != sha[f])]
        data_ok = Path(data.get('data_dir', '')).name == DATA_NAME
        out[name] = {'ok': cfg.get('name') == CONFIG and not mism and data_ok, 'config': cfg.get('name'),
                     'data_dir': data.get('data_dir'), 'sha256_mismatch': mism, 'sha256_at_start': sha}
    return out


def evaluate(e, u, s, x, seeds, rb):
    return {'G1': g1(e, seeds), 'G2': g2(u, seeds, rb), 'G3': g3(s, x, seeds), 'G4': g4(s, seeds), 'G5': g5(s, seeds)}


def main():
    E, ST = Path(sys.argv[1]), Path(sys.argv[2])
    e, u = pd.read_csv(E/'ensemble.csv'), pd.read_csv(E/'unlearning.csv')
    s, x = pd.read_csv(ST/'steps.csv'), pd.read_csv(ST/'exactness.csv')
    seeds = sorted(int(v) for v in e.seed.unique())
    for name, df in [('unlearning', u), ('steps', s), ('exactness', x)]:
        if sorted(int(v) for v in df.seed.unique()) != seeds: raise ValueError(f'{name}.csv seeds differ from ensemble.csv')
    cfg_p = E/'train_config.json'
    rb = json.loads(cfg_p.read_text())['fl']['rounds'] if cfg_p.exists() else int(u.rounds.max())
    gates = evaluate(e, u, s, x, seeds, rb)
    rest = [v for v in seeds if v != SEEN_SEED]
    sens = {k: v for k, v in evaluate(e, u, s, x, rest, rb).items() if k != 'G3'}
    integ = integrity({'ensemble': E, 'stream': ST})
    all_pass = all(g['pass'] for g in gates.values()); integ_ok = all(v['ok'] for v in integ.values())
    changed = [k for k in sens if sens[k]['pass'] != gates[k]['pass']]
    result = {'confirmed': all_pass and integ_ok, 'all_gates_pass': all_pass, 'integrity_ok': integ_ok, 'seeds': seeds,
              'thresholds': {'HIGH_FPR': HIGH_FPR, 'D_F1': D_F1, 'D_FPR': D_FPR}, 'gates': gates,
              'sensitivity_without_seed_%d' % SEEN_SEED: {'seeds': rest, 'gates': sens, 'verdict_changed': changed},
              'integrity': integ}

    print(f'seeds {seeds}; R_base {rb}; thresholds FPR>{HIGH_FPR} = high-FPR mode, dF1 >= -{D_F1}, dFPR <= +{D_FPR}')
    for k, g in gates.items():
        print(f'\n{k}: {"PASS" if g["pass"] else "FAIL"}   (without seed {SEEN_SEED}: {"PASS" if sens.get(k, g)["pass"] else "FAIL"})')
        if 'comparisons' in g: print(pd.DataFrame(g['comparisons']).round(4).to_string(index=False))
        for key in ('high_fpr', 'not_exact_at_some_step', 'not_bit_exact'):
            if g.get(key): print(f'  {key}: {g[key][:10]}{" ..." if len(g[key]) > 10 else ""} ({len(g[key])} total)')
        if k == 'G3': print(f'  coverage complete: {g["coverage_complete"]}')
        if k == 'G4': print(f'  reference over the stream: F1 min {g["F1_min"]:.4f}, FPR max {g["FPR_max"]:.4f}')
    print('\nintegrity:', {n: (v['ok'], v.get('config'), v.get('sha256_mismatch', v.get('reason'))) for n, v in integ.items()})
    print(f'verdict changed without seed {SEEN_SEED}: {changed or "none"}')
    print(f'\nCONFIRMED: {result["confirmed"]}  (all gates pass: {all_pass}; integrity ok: {integ_ok})')
    if len(sys.argv) > 3:
        Path(sys.argv[3]).write_text(json.dumps(result, indent=1, default=lambda o: o.item() if isinstance(o, np.generic) else str(o)))


if __name__ == '__main__':
    main()
