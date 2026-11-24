"""Sensitivity sweep of docs/dt_sensitivity_plan.md, run with the unchanged C1/C2 runner (scripts/dt/run_orchestration.py).

    python scripts/sens/run_sensitivity.py write          # write configs/sens/*.json (an existing different one is an error)
    python scripts/sens/run_sensitivity.py list           # configs, blocks and a time estimate
    python scripts/sens/run_sensitivity.py run [--only s1 s2 ...]

Every config is the C1/C2 config (configs/dt_orchestration.json) with the keys of its stage changed. Each config is one
run directory results/dt_sensitivity/<name>/ with its own protocol_lock.json (the plan hashed is the sensitivity plan).
A config whose complete.json exists is skipped; a started one is resumed (--resume). A failure is recorded in
runs.jsonl and the next config runs. This file lives outside scripts/dt so the C1/C2 lock does not see it.
"""
import argparse
import copy
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

BASE_CONFIG = ROOT/'configs/dt_orchestration.json'
PLAN = ROOT/'docs/dt_sensitivity_plan.md'
CONFIG_DIR = ROOT/'configs/sens'
OUT = ROOT/'results/dt_sensitivity'
RUNNER = ROOT/'scripts/dt/run_orchestration.py'
HIDDEN_RUNNER = ROOT/'scripts/sens/run_hidden.py'

ARMS4 = ['dt_queue', 'edf', 'rows_legacy', 'immediate']
ARMS3 = ['dt_queue', 'edf', 'rows_legacy']
ENVS = ['none', 'gradual_clients', 'jitter_bandwidth']
C1C2_SEEDS = [73042, 73043, 73044]           # same order => same repeat index => same streams and stress paths
DEV = dict(seeds=[75042, 75043], stream_seed_base=85000, stress_seed=95000)   # S4 only: selects values
MINUTES_PER_BLOCK = {0.2: 3.2, 0.5: 4.8, 1.0: 7.5}   # 4 arms + fit, measured in the C1/C2 run (K = 12)


def g(x):
    return f'{x:g}'


def stages():
    """name -> (stage, overrides). Overrides replace keys of the C1/C2 config."""
    out = {}
    for b in (0.5, 0.75, 1.0, 1.5):
        for d in (1.0, 1.5, 2.0, 3.0):
            out[f's1_b{g(b)}_d{g(d)}'] = ('s1', dict(budget_scale=b, deadline_scale=d, seeds=C1C2_SEEDS[:2],
                                                     loads=[0.2], environments=ENVS, arms=ARMS4))
    for f in (2.0, 8.0):
        out[f's2_slow{g(f)}'] = ('s2', dict(slow_factor_max=f, seeds=C1C2_SEEDS, loads=[0.2, 0.5],
                                            environments=['gradual_clients'], arms=ARMS4))
    for lo, hi in ((2.0, 8.0), (12.5, 50.0)):
        out[f's2_bw{g(lo)}-{g(hi)}'] = ('s2', dict(bandwidth_mbps_range=[lo, hi], seeds=C1C2_SEEDS, loads=[0.2, 0.5],
                                                   environments=['jitter_bandwidth'], arms=ARMS4))
    for k in (6, 24):
        out[f's3_k{k}'] = ('s3', dict(requests=k, seeds=C1C2_SEEDS[:2], loads=[0.2, 0.5], environments=ENVS,
                                      arms=ARMS4))
    s4 = dict(DEV, loads=[0.2], environments=['gradual_clients', 'jitter_bandwidth'], arms=ARMS3)
    out['s4_ref'] = ('s4', dict(s4))
    for key, short, values in (('margin_q', 'q', (0.8, 0.95)), ('margin_window', 'w', (5, 20)),
                               ('expert_eta', 'eta', (1.0, 10.0)), ('expert_floor', 'floor', (0.02, 0.25)),
                               ('alpha', 'alpha', (0.1, 0.5))):
        for v in values:
            out[f's4_{short}{g(v)}'] = ('s4', dict(s4, **{key: v}))
    for h in ((32, 16), (128, 64)):
        out[f's5_h{h[0]}-{h[1]}'] = ('s5', dict(hidden=list(h), seeds=C1C2_SEEDS[:2], loads=[0.2], environments=ENVS,
                                                arms=ARMS4))
    return out


def build(name, stage, overrides, base):
    cfg = copy.deepcopy(base)
    for key in ('c1_baselines', 'c2_ablations', 'analysis'):
        cfg.pop(key)
    changed = {k: v for k, v in overrides.items() if base.get(k) != v}
    cfg.update(copy.deepcopy(overrides))
    cfg['purpose'] = f'Sensitivity {stage} ({name}); docs/dt_sensitivity_plan.md. Descriptive, not a confirmation.'
    cfg['sensitivity'] = dict(name=name, stage=stage, changed_from_c1c2=changed, base_config='configs/dt_orchestration.json')
    return cfg


def lf_sha256(path):
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def blocks_of(cfg):
    return len(cfg['seeds'])*len(cfg['families'])*len(cfg['loads'])*len(cfg['environments'])


def minutes_of(cfg):
    per = sum(MINUTES_PER_BLOCK[l] for l in cfg['loads'])/len(cfg['loads'])
    return blocks_of(cfg)*per*cfg['requests']/12*len(cfg['arms'])/4


def write():
    base = json.loads(BASE_CONFIG.read_text(encoding='utf-8'))
    CONFIG_DIR.mkdir(parents=True, exist_ok=True)
    for name, (stage, overrides) in stages().items():
        text = json.dumps(build(name, stage, overrides, base), indent=2)+'\n'
        path = CONFIG_DIR/f'{name}.json'
        if path.exists():
            if json.loads(path.read_text(encoding='utf-8')) != json.loads(text):
                raise ValueError(f'{path} differs from the plan; the sweep is frozen')
            continue
        path.write_text(text, encoding='utf-8')
    print(f'{len(stages())} configs in {CONFIG_DIR.relative_to(ROOT)}')


def configs(only=None):
    out = []
    for name, (stage, _) in stages().items():
        if only and stage not in only:
            continue
        path = CONFIG_DIR/f'{name}.json'
        out.append((name, stage, path, json.loads(path.read_text(encoding='utf-8'))))
    return out


def show(only=None):
    total = 0.
    for name, stage, _, cfg in configs(only):
        m = minutes_of(cfg)
        total += m
        print(f'{name:22s} {stage}  blocks={blocks_of(cfg):3d}  ~{m/60:4.1f} h  {cfg["sensitivity"]["changed_from_c1c2"]}')
    print(f'total ~{total/60:.1f} h')


def append(record):
    with (OUT/'runs.jsonl').open('a', encoding='utf-8') as handle:
        handle.write(json.dumps(record)+'\n')


def run(only=None):
    from scripts.dt.run_predictive_confirmation import keep_awake
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT/'logs').mkdir(exist_ok=True)
    selected = configs(only)
    manifest = OUT/'manifest.json'
    files = [PLAN, Path(__file__), HIDDEN_RUNNER, RUNNER] + [p for _, _, p, _ in configs()]
    current = {p.resolve().relative_to(ROOT).as_posix(): lf_sha256(p) for p in files}
    if manifest.exists():
        frozen = json.loads(manifest.read_text(encoding='utf-8'))['sha256_lf']
        changed = sorted(k for k in set(frozen) | set(current) if frozen.get(k) != current.get(k))
        if changed:
            raise ValueError(f'sweep files changed since the first run: {changed}')
    else:
        head = subprocess.run(['git', 'rev-parse', 'HEAD'], cwd=ROOT, capture_output=True, text=True).stdout.strip()
        manifest.write_text(json.dumps(dict(sha256_lf=current, git_head=head, python=sys.executable,
                                            first_run_local=time.strftime('%Y-%m-%dT%H:%M:%S')), indent=1),
                            encoding='utf-8')
    with keep_awake():
        for name, stage, path, cfg in selected:
            out = OUT/name
            if (out/'complete.json').exists():
                continue
            runner = HIDDEN_RUNNER if 'hidden' in cfg else RUNNER
            cmd = [sys.executable, str(runner), str(out), '--config', str(path), '--plan', str(PLAN)]
            if out.exists():
                cmd.append('--resume')
            started = time.strftime('%Y-%m-%dT%H:%M:%S')
            print(f'[{started}] {name} start', flush=True)
            with (OUT/'logs'/f'{name}.log').open('a', encoding='utf-8') as log:
                code = subprocess.run(cmd, cwd=ROOT, stdout=log, stderr=subprocess.STDOUT).returncode
            ended = time.strftime('%Y-%m-%dT%H:%M:%S')
            append(dict(name=name, stage=stage, started_local=started, ended_local=ended, exit_code=code,
                        resumed='--resume' in cmd))
            print(f'[{ended}] {name} exit={code}', flush=True)


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('action', choices=['write', 'list', 'run'])
    parser.add_argument('--only', nargs='*', choices=['s1', 's2', 's3', 's4', 's5'])
    args = parser.parse_args(argv)
    if args.action == 'write':
        write()
    elif args.action == 'list':
        show(args.only)
    else:
        run(args.only)


if __name__ == '__main__':
    main()
