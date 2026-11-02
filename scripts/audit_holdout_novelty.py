"""Measure how new each holdout split is relative to train: nearest-neighbour distance in encoded space.

The exact-hash overlap audit cannot see rows that differ only by float jitter. Output goes to a
separate file so the processed data directory stays immutable.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from sklearn.neighbors import NearestNeighbors
from src.data.prepare import digest, write_json


def novelty(train_x, train_y, x, y):
    distance, index = NearestNeighbors(n_neighbors=1).fit(train_x).kneighbors(x)
    distance, index = distance[:, 0], index[:, 0]
    return {'rows': len(y), 'distance_quantiles': {str(q): float(np.quantile(distance, q)) for q in [.1, .25, .5, .75, .9]},
            'fraction_within': {str(t): float((distance <= t).mean()) for t in [0., 1e-6, 1e-3, 1e-2, 1e-1]},
            'nearest_train_label_agreement': float((train_y[index] == y).mean())}


def audit(data, output):
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    manifest = json.loads((data/'manifest.json').read_text())
    for name, expected in manifest['artifacts_sha256'].items():
        if digest(data/name) != expected:
            raise ValueError(f'Data artifact hash mismatch: {name}')
    arrays = {s: dict(np.load(data/f'{s}.npz', allow_pickle=False)) for s in ['train', 'validation', 'calibration', 'test']}
    result = {'data': str(data), 'data_manifest_sha256': digest(data/'manifest.json'),
              'splits': {s: novelty(arrays['train']['X'], arrays['train']['y'], arrays[s]['X'], arrays[s]['y'])
                         for s in ['validation', 'calibration', 'test']}}
    output.parent.mkdir(parents=True, exist_ok=True)
    write_json(output, result)
    return result


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(audit(args.data, args.output), indent=2))
