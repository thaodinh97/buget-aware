"""Load a preprocessed split directory after checking its audit gate, artifact hashes and split disjointness."""
import json

import numpy as np

from src.data.prepare import digest


def load_data(path):
    audit = json.loads((path/'audit.json').read_text())
    if not audit['training_ready']:
        raise ValueError('Preprocessing gate is closed')
    if 'cross_split_encoded_overlap' not in audit or any(audit['cross_split_encoded_overlap'].values()):
        raise ValueError('Missing/failed encoded feature leakage audit; regenerate data')
    manifest = json.loads((path/'manifest.json').read_text())
    for name, expected in manifest['artifacts_sha256'].items():
        if digest(path/name) != expected:
            raise ValueError(f'Data artifact hash mismatch: {name}')
    data = {}
    for split in ['train', 'validation', 'calibration', 'test']:
        with np.load(path/f'{split}.npz', allow_pickle=False) as arrays:
            data[split] = dict(arrays)
    for a in data:
        for b in data:
            if a < b and (set(data[a]['row_id']) & set(data[b]['row_id']) or set(data[a]['group_id']) & set(data[b]['group_id'])):
                raise ValueError('Overlapping split IDs/groups')
    return data
