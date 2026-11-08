"""Shared pieces of the v2 shard protocol (docs/shard_v2_plan.md). Importable, no side effects.

- DESIGNS: the v2 design (seeds, numbers of shards, client constraints) and two smoke designs that are never evidence;
- assign / check_design / structure_plan: shards of >= min_clients_design clients; during a deletion stream a shard left
  with fewer than merge_below present clients is merged into the active shard with the fewest present clients (ties: lowest
  index). The plan depends only on the request stream (metadata), so it is the same for every policy and seed;
- data_content_sha256: hash of the array contents actually used (train, validation, calibration), independent of the
  path, the platform and the npz container;
- norm_rel: POSIX path relative to the repo root (case-insensitive drive letter on Windows), else absolute POSIX;
- start_run: loads the data, checks its content hash against the design, writes OUT/run.json;
- noninferior: one-sided paired-t noninferiority test over training seeds.
"""
import hashlib, json, os
from pathlib import Path
import numpy as np
from scipy import stats

ROOT = Path(__file__).resolve().parents[2]
DATA_DEFAULT = 'data/processed/connected_v3_auxfit_seed42'
# array content of connected_v3_auxfit_seed42 (train, validation, calibration), fixed before the v2 runs
DATA_SHA256 = 'f130057e20f1d9eb0857dac5b8ce90c9f0f16b544a8d4b5675b88f24eea0ef3a'
SMOKE_CONFIG = {'fl': {'rounds': 3, 'local_epochs': 1, 'batch_size': 512, 'lr': 1e-2},
                'unlearn_budgets': [1, 3], 'shard_fixed_rounds': 2, 'backfill_rounds': 2}
DESIGNS = {
    'v2': {'train_config': 'standard', 'seeds': list(range(50042, 50052)), 'shards_run1': [1, 2, 4, 5], 'shards_run2': [4, 5],
           'min_clients_design': 4, 'merge_below': 3},
    # smoke designs: plumbing checks only, never evidence; seeds 7, 8 are not v2 seeds. 'smoke' mirrors the v2 structure
    # (so the evaluator runs end to end); 'smoke_empty' (2 clients per shard, no merging) exercises the empty-shard removal
    'smoke': {'train_config': 'smoke', 'seeds': [7, 8], 'shards_run1': [1, 2, 4, 5], 'shards_run2': [4, 5],
              'min_clients_design': 4, 'merge_below': 3},
    'smoke_empty': {'train_config': 'smoke', 'seeds': [7, 8], 'shards_run1': [1, 10], 'shards_run2': [10],
                    'min_clients_design': 2, 'merge_below': 0},
}
TRACKED = ['docs/shard_v2_plan.md', 'scripts/shard/evaluate_shard_v2.py', 'scripts/shard/shard_v2.py',
           'scripts/shard/train_config.py', 'scripts/shard/run_shard_ensemble_check_v2.py', 'scripts/shard/run_shard_stream_v2.py']


def text_sha256(path):
    """SHA-256 of a text file with CRLF normalised to LF (survives git's line-ending conversion)."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def norm_rel(path, root=ROOT):
    """POSIX path relative to `root` when inside it (drive letter / case differences on Windows ignored), else absolute POSIX."""
    p, r = Path(os.path.abspath(path)), Path(os.path.abspath(root))
    a, b = os.path.normcase(str(p)), os.path.normcase(str(r))
    if a == b: return '.'
    if a.startswith(b.rstrip(os.sep) + os.sep): return Path(str(p)[len(str(r).rstrip(os.sep)) + 1:]).as_posix()
    return p.as_posix()


def data_content_sha256(data, splits=('train', 'validation', 'calibration')):
    """Hash of every array of the given splits: key, little-endian dtype, shape and bytes, in sorted key order."""
    h = hashlib.sha256()
    for split in splits:
        for key in sorted(data[split]):
            a = np.asarray(data[split][key]); le = a.dtype.newbyteorder('<')
            a = np.ascontiguousarray(a.astype(le, copy=False))
            h.update(f'{split}/{key}/{le.str}/{a.shape};'.encode()); h.update(a.tobytes())
    return h.hexdigest()


def assign(S, clients=20, seed=7):
    perm = np.random.default_rng(seed).permutation(clients)
    return [sorted(perm[j::S].tolist()) for j in range(S)]


def check_design(assignment, min_clients):
    sizes = [len(a) for a in assignment]
    if min(sizes) < min_clients: raise ValueError(f'shard sizes {sizes} violate the minimum of {min_clients} clients per shard')
    return sizes


def structure_plan(assignment, stream, merge_below):
    """Per request: shard structure before/after, which shard (re)trains, which shards leave the ensemble, and the event."""
    cur = [list(a) for a in assignment]; plan = []
    for r in stream:
        before = [list(a) for a in cur]; c = r['client']
        j = next(i for i, a in enumerate(cur) if c in a); event, retrain, retired = 'retrain', j, []
        if r['kind'] == 'client':
            cur[j].remove(c)
            others = [i for i, a in enumerate(cur) if a and i != j]
            if not cur[j]: event, retrain, retired = 'removed_empty', None, [j]
            elif len(cur[j]) < merge_below and others:
                t = min(others, key=lambda i: (len(cur[i]), i))
                cur[t] = sorted(cur[t] + cur[j]); cur[j] = []; event, retrain, retired = f'merge_{j}_into_{t}', t, [j]
        plan.append({'step': r['step'], 'client': int(c), 'kind': r['kind'], 'event': event, 'retrain': retrain, 'retired': retired,
                     'assign_before': before, 'assign_after': [list(a) for a in cur]})
    return plan


def start_run(out_dir, runner_file):
    """Load the data of the design, check its content hash, write OUT/run.json. Returns (design_name, design, config, data)."""
    from scripts.run_pilot import load_data
    from scripts.shard.train_config import CONFIGS
    name = os.environ.get('SHARD_DESIGN', 'v2')
    if name not in DESIGNS: raise ValueError(f'SHARD_DESIGN must be one of {sorted(DESIGNS)}, got {name!r}')
    design = DESIGNS[name]; cfg = SMOKE_CONFIG if design['train_config'] == 'smoke' else CONFIGS[design['train_config']]
    runner = norm_rel(runner_file)
    if runner not in TRACKED: raise ValueError(f'runner {runner} is not one of the tracked v2 runners')
    data_dir = Path(os.environ.get('SHARD_DATA_DIR', DATA_DEFAULT))
    if not data_dir.is_absolute(): data_dir = ROOT/data_dir
    missing = [f for f in TRACKED if not (ROOT/f).exists()]
    if missing and name == 'v2': raise FileNotFoundError(f'tracked files missing before a v2 run: {missing}')
    data = load_data(data_dir); content = data_content_sha256(data)
    if content != DATA_SHA256: raise ValueError(f'data content {content} != expected {DATA_SHA256} ({data_dir})')
    record = {'design_name': name, 'design': design, 'train_config_name': design['train_config'], 'train_config': cfg,
              'runner': runner, 'data_dir': norm_rel(data_dir), 'data_dir_abs': Path(os.path.abspath(data_dir)).as_posix(),
              'data_content_sha256': content,
              'manifest_artifacts_sha256': json.loads((data_dir/'manifest.json').read_text())['artifacts_sha256'],
              'sha256_at_start': {f: text_sha256(ROOT/f) for f in TRACKED if f not in missing}}
    (Path(out_dir)/'run.json').write_text(json.dumps(record, indent=1))
    return name, design, cfg, data


def noninferior(diff, margin, worse, alpha=.05):
    """One-sided paired-t noninferiority over seeds. diff = candidate - reference per seed.
    worse='lower' (macro-F1): pass iff mean - t_{1-alpha,n-1} * se > -margin;
    worse='higher' (FPR):     pass iff mean + t_{1-alpha,n-1} * se <  margin."""
    d = np.asarray(diff, float)
    if d.ndim != 1 or len(d) < 2 or not np.isfinite(d).all(): raise ValueError('need >= 2 finite paired differences')
    n = len(d); se = d.std(ddof=1)/np.sqrt(n); t = stats.t.ppf(1 - alpha, n - 1)
    if worse == 'lower': bound = d.mean() - t*se; ok = bound > -margin
    elif worse == 'higher': bound = d.mean() + t*se; ok = bound < margin
    else: raise ValueError(worse)
    return {'mean': float(d.mean()), 'bound': float(bound), 'se': float(se), 'n': n, 'ok': bool(ok)}
