"""Frozen inputs and startup records for the v3 shard protocol.

The v1/v2 startup helpers are deliberately left unchanged. Only train, validation
and calibration are loaded here; the test split is never opened.
"""
import copy
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import pandas as pd
import scipy
import sklearn
import torch

from scripts.shard.shard_v2 import (ROOT, DATA_DEFAULT, DATA_SHA256, SMOKE_CONFIG,
                                   data_content_sha256, norm_rel, text_sha256)
from scripts.shard.stream_v3 import FAMILIES, STREAM
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition

RUNNER = 'scripts/shard/run_shard_stream_v3.py'
METADATA_DEFAULT = 'results/shard_v2/ensemble_check_20c/meta.json'
EXPECTED_CLIENT_ROWS = [4768, 1308, 6095, 6102, 3364, 4508, 2490, 12280, 1056, 4445,
                        29317, 4278, 15035, 7034, 814, 1794, 85, 5144, 317, 4050]
DESIGN = {'train_config': 'standard', 'seeds': list(range(60042, 60052)), 'shards': [4, 5],
          'clients': 20, 'partition_seed': 42, 'partition_alpha': .5, 'assignment_seed': 7,
          'min_clients_design': 4, 'merge_below': 3, 'reference_steps': [5, 10, 15, 20]}
DESIGNS = {'v3': DESIGN, 'smoke_v3': {**DESIGN, 'train_config': 'smoke', 'seeds': [7, 8]}}
TRACKED_V3 = ['docs/shard_v3_plan.md', 'scripts/shard/stream_v3.py', RUNNER,
              'scripts/shard/evaluate_shard_v3.py', 'scripts/shard/shard_v3.py',
              'scripts/shard/shard_v2.py', 'scripts/shard/train_config.py',
              'src/fu/shard_exact.py', 'src/fu/shard_cost.py']


def write_json(path, value):
    def convert(obj):
        if isinstance(obj, np.generic):
            return obj.item()
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, Path):
            return obj.as_posix()
        raise TypeError(type(obj).__name__)
    Path(path).write_text(json.dumps(value, indent=1, default=convert), encoding='utf-8')


def source_hashes():
    missing = [name for name in TRACKED_V3 if not (ROOT/name).is_file()]
    if missing:
        raise FileNotFoundError(f'tracked files missing before a v3 run: {missing}')
    return {name: text_sha256(ROOT/name) for name in TRACKED_V3}


def machine_info(device='cpu'):
    return {'hostname': platform.node(), 'platform': platform.platform(), 'machine': platform.machine(),
            'processor': platform.processor(), 'python': platform.python_version(), 'numpy': np.__version__,
            'pandas': pd.__version__, 'scipy': scipy.__version__, 'sklearn': sklearn.__version__,
            'torch': str(torch.__version__), 'threads': torch.get_num_threads(), 'device': str(device)}


def load_protocol_data(data_dir):
    """Check preprocessing provenance and load the three allowed splits only."""
    data_dir = Path(data_dir)
    audit = json.loads((data_dir/'audit.json').read_text(encoding='utf-8'))
    if not audit.get('training_ready'):
        raise ValueError('Preprocessing gate is closed')
    overlap = audit.get('cross_split_encoded_overlap')
    if not isinstance(overlap, dict) or any(overlap.values()):
        raise ValueError('Missing/failed encoded feature leakage audit; regenerate data')
    manifest = json.loads((data_dir/'manifest.json').read_text(encoding='utf-8'))
    data = {}
    for name in ('train', 'validation', 'calibration'):
        filename = f'{name}.npz'
        expected = manifest['artifacts_sha256'].get(filename)
        if expected is None or hashlib.sha256((data_dir/filename).read_bytes()).hexdigest() != expected:
            raise ValueError(f'Data artifact hash mismatch: {filename}')
        with np.load(data_dir/filename, allow_pickle=False) as arrays:
            data[name] = dict(arrays)
    for a in data:
        for b in data:
            if a < b and any(set(data[a][key]) & set(data[b][key]) for key in ('row_id', 'group_id')):
                raise ValueError('Overlapping split IDs/groups')
    return data, manifest


def run_selection():
    """Design name and family of this process, from SHARD_DESIGN and SHARD_V3_FAMILY.

    Neither variable has a default: a forgotten one must not start the confirmatory
    design, nor write one family's stream under another family's directory.
    """
    name = os.environ.get('SHARD_DESIGN')
    if name not in DESIGNS:
        raise ValueError(f'SHARD_DESIGN must be set to one of {sorted(DESIGNS)}, got {name!r}')
    family = os.environ.get('SHARD_V3_FAMILY')
    if family not in FAMILIES:
        raise ValueError(f'SHARD_V3_FAMILY must be set to one of {FAMILIES}, got {family!r}')
    return name, family


def start_run(out_dir, runner_file=ROOT/RUNNER):
    """Validate the frozen design and write run.json before training.

    Returns (design_name, design, train_config, data, parts, run_record).
    SHARD_DESIGN=smoke_v3 uses short training, and is never confirmatory evidence.
    """
    name, family = run_selection()
    runner = norm_rel(runner_file)
    if runner != RUNNER:
        raise ValueError(f'runner {runner} is not the tracked v3 runner')
    hashes = source_hashes()
    design = copy.deepcopy(DESIGNS[name])
    cfg = copy.deepcopy(SMOKE_CONFIG if name == 'smoke_v3' else CONFIGS['standard'])
    data_dir = Path(os.environ.get('SHARD_DATA_DIR', DATA_DEFAULT))
    if not data_dir.is_absolute():
        data_dir = ROOT/data_dir
    data, manifest = load_protocol_data(data_dir)
    content = data_content_sha256(data)
    if content != DATA_SHA256:
        raise ValueError(f'data content {content} != expected {DATA_SHA256} ({data_dir})')
    metadata_path = ROOT/METADATA_DEFAULT
    metadata = json.loads(metadata_path.read_text(encoding='utf-8'))
    if metadata.get('client_rows') != EXPECTED_CLIENT_ROWS or metadata.get('partition_seed') != DESIGN['partition_seed']:
        raise ValueError('design metadata differs from the frozen v3 client counts/partition seed')
    parts = partition(data['train']['y'], design['partition_seed'], design['partition_alpha'], design['clients'])
    client_rows = [len(p) for p in parts]
    if client_rows != EXPECTED_CLIENT_ROWS:
        raise ValueError(f'client row counts {client_rows} != frozen metadata {EXPECTED_CLIENT_ROWS}')
    device = os.environ.get('FL_DEVICE', 'cpu')
    if device != 'cpu':
        raise ValueError('v3 requires FL_DEVICE=cpu for the fixed deterministic protocol')
    record = {'design_name': name, 'design': design, 'family': family,
              'train_config_name': design['train_config'], 'train_config': cfg,
              'runner': runner, 'data_dir': norm_rel(data_dir), 'data_dir_abs': data_dir.resolve().as_posix(),
              'data_content_sha256': content, 'manifest_artifacts_sha256': manifest['artifacts_sha256'],
              'sha256_at_start': hashes, 'client_rows': client_rows, 'partition_seed': design['partition_seed'],
              'metadata_file': METADATA_DEFAULT, 'metadata_sha256': text_sha256(metadata_path),
              'stream_config': copy.deepcopy(STREAM), 'machine': machine_info(device),
              'reproducible': False, 'reproducibility_checks': [], 'completed': False}
    write_json(Path(out_dir)/'run.json', record)
    return name, design, cfg, data, parts, record
