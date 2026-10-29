"""Create one read-only feature/label shard for each frozen FL client partition."""
import argparse
import hashlib
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.run_pilot import load_data


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def prepare(data_path, source, output):
    if output.exists() and any(output.iterdir()):
        raise FileExistsError(f'Refusing overwrite: {output}')
    data = load_data(data_path)
    if sha256(data_path / 'manifest.json') != json.loads((source / 'manifest.json').read_text())['data_manifest_sha256']:
        raise ValueError('Source run was produced from a different frozen dataset')
    for name, expected_hash in json.loads((source / 'artifact_hashes.json').read_text()).items():
        artifact = source / Path(name.replace('\\', '/'))
        if not artifact.is_file():
            raise ValueError(f'Missing source artifact: {name}')
        actual_hash = sha256(artifact)
        if actual_hash != expected_hash and name != 'metrics.csv':
            raise ValueError(f'Modified source artifact: {name}')
    train = data['train']
    with np.load(source / 'partitions.npz', allow_pickle=False) as archive:
        partitions = [archive[f'client_{i}'] for i in range(5)]
    index = {row_id: i for i, row_id in enumerate(train['row_id'])}
    output.mkdir(parents=True, exist_ok=True)
    clients = []
    for client_id, row_ids in enumerate(partitions):
        if len(row_ids) == 0 or len(set(row_ids)) != len(row_ids):
            raise ValueError(f'Invalid frozen partition for client {client_id}')
        indices = np.array([index[row_id] for row_id in row_ids], dtype=np.int64)
        path = output / f'client_{client_id}.npz'
        np.savez_compressed(path, X=train['X'][indices], y=train['y'][indices])
        clients.append({'client_id': client_id, 'samples': len(indices), 'sha256': sha256(path)})
    if sum(client['samples'] for client in clients) != len(train['y']):
        raise AssertionError('Client shards do not cover the training split')
    manifest = {
        'clients': clients,
        'sample_total': len(train['y']),
        'input_dim': int(train['X'].shape[1]),
        'source_manifest_sha256': sha256(source / 'manifest.json'),
        'data_manifest_sha256': sha256(data_path / 'manifest.json'),
        'privacy_note': 'Local Docker demo shards; no DP, encryption, or secure aggregation.',
    }
    (output / 'manifest.json').write_text(json.dumps(manifest, indent=2), encoding='utf-8')
    return manifest


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, default=Path('data/processed/connected_v3_seed42'))
    parser.add_argument('--source', type=Path, default=Path('runs/pilot_v3_seed42'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(prepare(args.data, args.source, args.output), indent=2), flush=True)
