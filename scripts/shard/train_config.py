"""Training configuration of the shard direction, shared by run_shard_ensemble_check.py and run_shard_stream.py.

Selected with the environment variable SHARD_TRAIN_CONFIG (default 'standard'):
- 'standard': the standard training config of the shard direction, fixed in docs/shard_standard_config_plan.md before the
  confirmation runs. It is the DynamicFU paper's training (Adam lr 1e-2, R_base = 20 rounds, 2 local epochs, batch 512),
  as in run_shard_paper_scale.py, with the reduced budgets scaled to R_base the same way as there.
- 'repo': the repo FL config (Adam lr 1e-3, 15 rounds, 1 local epoch, batch 512) used by the exploration runs in
  results/shard_exploration*/. Set SHARD_TRAIN_CONFIG=repo to reproduce them.

load() also writes OUT/train_config.json with the config and the SHA-256 (line endings normalised to LF) of the protocol,
the gate evaluator, this module and the calling runner, as they were when the run started.
"""
import copy, hashlib, json, os, sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
CONFIGS = {
    'standard': {'fl': {'rounds': 20, 'local_epochs': 2, 'batch_size': 512, 'lr': 1e-2},
                 'unlearn_budgets': [1, 2, 5, 10, 20], 'shard_fixed_rounds': 10, 'backfill_rounds': 7},
    'repo': {'fl': {'rounds': 15, 'local_epochs': 1, 'batch_size': 512, 'lr': 1e-3},
             'unlearn_budgets': [1, 2, 5, 10, 15], 'shard_fixed_rounds': 7, 'backfill_rounds': 5},
}
TRACKED = ['docs/shard_standard_config_plan.md', 'scripts/shard/evaluate_standard_config.py', 'scripts/shard/train_config.py']


def text_sha256(path):
    """SHA-256 of a text file with CRLF normalised to LF, so the hash survives git's line-ending conversion."""
    return hashlib.sha256(Path(path).read_bytes().replace(b'\r\n', b'\n')).hexdigest()


def load(out_dir):
    name = os.environ.get('SHARD_TRAIN_CONFIG', 'standard')
    if name not in CONFIGS: raise ValueError(f'SHARD_TRAIN_CONFIG must be one of {sorted(CONFIGS)}, got {name!r}')
    cfg = copy.deepcopy(CONFIGS[name])
    runner = Path(sys.argv[0]).resolve()
    files = TRACKED + ([runner.relative_to(ROOT).as_posix()] if runner.is_relative_to(ROOT) else [])
    sha = {f: text_sha256(ROOT/f) for f in files if (ROOT/f).exists()}
    (Path(out_dir)/'train_config.json').write_text(json.dumps({'name': name, **cfg, 'sha256_at_start': sha}, indent=1))
    return name, cfg
