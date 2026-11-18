"""Re-encode a prepared split directory with log1p numeric columns appended, keeping its split exactly.

    python src/data/prepare_log1p.py --source data/processed/connected_v3_auxfit_seed42 \
        --config configs/data_connected_auxfit_log1p.json --output data/processed/connected_v3_auxfit_log1p_seed42

Encoding: [base z-scores | standardized log1p | base one-hot]. Every numeric predictor keeps its z-score column (median
fill, mean/scale) and gets a second column, log1p of the filled value standardized; all statistics come from the
transform-fit split (validation for "validation_only"). Without the log, the z-scores of heavy-tailed byte counts are
nearly equal for every small flow (docs/fpr_threshold_results.md). A log-only column is avoided because in float32 it
merges very large flows that differ by tens of bytes.

The split is NOT derived again. Rows, order, split, groups, labels and row ids are those of the source directory, whose
artifacts are verified first; features are re-encoded from the raw CSV (same file hash as the source). Deriving the split
again with src/data/prepare.py changes it: the leakage gate refines groups on the encoded vectors, and other encoded
vectors give other refinements. Here the gate is only checked: if two splits share an encoded vector, nothing is written.
src/data/prepare.py itself is reused for its helpers and stays byte-identical (it is hash-locked by
results/dt_*_confirmation/protocol_lock.json). check_numeric_transform() verifies the transform of a prepared directory.
"""
import argparse
import hashlib
import json
import platform
import shutil
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd

import src.data.prepare as base

TRANSFORMS = ('append_log1p',)
SPLITS = ('train', 'validation', 'calibration', 'test')
MAY_DIFFER = {'preprocessing_version', 'numeric_transform'}


def _log1p(values):
    negative = values.columns[(values < 0).any()].tolist()
    if negative:
        raise ValueError(f'log1p needs non-negative numeric predictors; negative values in {negative}')
    return np.log1p(values)


def fit_transformer(fit, numeric, categorical):
    """base.fit_transformer, plus mean and scale of log1p(filled values) and the names of the appended columns."""
    state = base.fit_transformer(fit, numeric, categorical)
    values = _log1p(base.numeric_frame(fit, numeric).fillna(state['median']))
    state['log_mean'] = values.mean().to_dict()
    state['log_scale'] = values.std(ddof=0).replace(0, 1).to_dict()
    k = len(numeric)
    names = state['feature_names']
    state['feature_names'] = names[:k] + [f'log1p({c})' for c in numeric] + names[k:]
    state['numeric_transform'] = 'append_log1p'
    return state


def transform(frame, state):
    """[base z-scores | standardized log1p | base one-hot]."""
    if state.get('numeric_transform') != 'append_log1p':
        raise ValueError('state was not fitted with the append_log1p numeric transform')
    output = base.transform(frame, state)
    numeric = state['numeric']
    k = len(numeric)
    values = _log1p(base.numeric_frame(frame, numeric).fillna(state['median']))
    logs = ((values - pd.Series(state['log_mean'])) / pd.Series(state['log_scale'])).to_numpy(dtype=np.float32)
    output = np.concatenate([output[:, :k], logs, output[:, k:]], axis=1)
    if not np.isfinite(output).all():
        raise ValueError('Non-finite transformed values')
    return output


def reencode(source_dir, input_path, config_path, output_dir):
    source_dir, output_dir = Path(source_dir), Path(output_dir)
    if output_dir.exists():
        raise FileExistsError(f'Refusing to overwrite run: {output_dir}')
    config = json.loads(Path(config_path).read_text(encoding='utf-8'))
    if config.get('numeric_transform') not in TRANSFORMS:
        raise ValueError(f'config needs "numeric_transform" in {TRANSFORMS}')
    manifest = json.loads((source_dir/'manifest.json').read_text(encoding='utf-8'))
    audit = json.loads((source_dir/'audit.json').read_text(encoding='utf-8'))
    for name, expected in manifest['artifacts_sha256'].items():
        if base.digest(source_dir/name) != expected:
            raise ValueError(f'source artifact hash mismatch: {name}')
    if not audit['training_ready']:
        raise ValueError('source preprocessing gate is closed')
    differing = sorted(k for k in set(config) | set(manifest['config'])
                       if k not in MAY_DIFFER and config.get(k) != manifest['config'].get(k))
    if differing:
        raise ValueError(f'config differs from the source run beyond {sorted(MAY_DIFFER)}: {differing}')
    if base.digest(input_path) != manifest['dataset_sha256']:
        raise ValueError('raw dataset differs from the one the source was prepared from')
    numeric, categorical = config['numeric_features'], config['categorical_features']
    raw = pd.read_csv(input_path, low_memory=False)
    meta = pd.read_csv(source_dir/'metadata.csv')
    frame = raw.loc[[int(r.rsplit(':', 1)[1]) for r in meta.row_id]]
    splits = meta.split.to_numpy()
    fit_split = base.TRANSFORM_FIT[config['transform_fit']]
    state = fit_transformer(frame.loc[splits == fit_split], numeric, categorical)
    encoded, kept = {}, {}
    for name in SPLITS:
        mask = splits == name
        with np.load(source_dir/f'{name}.npz', allow_pickle=False) as arrays:
            kept[name] = {key: arrays[key] for key in ('y', 'row_id', 'group_id')}
        part = frame.loc[mask]
        if (not np.array_equal(kept[name]['row_id'], meta.row_id.to_numpy()[mask])
                or not np.array_equal(kept[name]['y'], part.label.to_numpy(dtype=np.int64))):
            raise ValueError(f'{name}: source rows or labels do not match metadata and raw data')
        encoded[name] = transform(part, state)
    signatures = np.concatenate([[hashlib.sha256(row.tobytes()).hexdigest() for row in encoded[n]] for n in SPLITS])
    overlap = base.overlaps(signatures, np.concatenate([np.full(len(encoded[n]), n) for n in SPLITS]))
    if any(overlap.values()):
        raise ValueError(f'encoded vectors shared across splits {overlap}; the split is kept, so nothing is written')
    output_dir.mkdir(parents=True)
    for name in SPLITS:
        np.savez_compressed(output_dir/f'{name}.npz', X=encoded[name], **kept[name])
    for name in ('metadata.csv', 'duplicate_mapping.csv'):
        shutil.copyfile(source_dir/name, output_dir/name)
    base.write_json(output_dir/'transformer.json', state)
    splits_audit = {n: {**audit['splits'][n], 'feature_dimension': int(encoded[n].shape[1])} for n in SPLITS}
    new_audit = {**audit, 'splits': splits_audit, 'cross_split_encoded_overlap': overlap,
                 'reencoded_from': source_dir.as_posix(), 'numeric_transform': config['numeric_transform']}
    base.write_json(output_dir/'audit.json', new_audit)
    new_manifest = {'config': config, 'dataset_path': str(Path(input_path).resolve()),
                    'dataset_sha256': manifest['dataset_sha256'], 'source_sha256': base.digest(base.__file__),
                    'extension_source_sha256': base.digest(__file__), 'config_sha256': base.digest(config_path),
                    'numeric_transform': config['numeric_transform'],
                    'reencoded_from': {'dir': source_dir.as_posix(), 'config': manifest['config'],
                                       'manifest_sha256': base.digest(source_dir/'manifest.json'),
                                       'artifacts_sha256': manifest['artifacts_sha256']},
                    'python': platform.python_version(), 'platform': platform.platform(),
                    'numpy': np.__version__, 'pandas': pd.__version__,
                    'transform_fit_rows_sha256': hashlib.sha256('\n'.join(meta.row_id[splits == fit_split]).encode()).hexdigest(),
                    'artifacts_sha256': {p.name: base.digest(p) for p in sorted(output_dir.iterdir())}}
    base.write_json(output_dir/'manifest.json', new_manifest)
    return new_audit


def check_numeric_transform(data_dir, expected):
    """Raise unless the prepared directory was encoded with the expected numeric transform ('none' = plain z-score)."""
    state = json.loads((Path(data_dir)/'transformer.json').read_text(encoding='utf-8'))
    got = state.get('numeric_transform', 'none')
    if got != expected:
        raise ValueError(f'{data_dir}: numeric transform {got!r}, expected {expected!r}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--source', required=True, help='prepared directory whose split is kept')
    parser.add_argument('--input', default='dataset/train_test_network.csv')
    parser.add_argument('--config', required=True)
    parser.add_argument('--output', required=True)
    args = parser.parse_args()
    result = reencode(args.source, args.input, args.config, args.output)
    print(json.dumps({k: result[k] for k in ('training_ready', 'cross_split_encoded_overlap', 'numeric_transform')}, indent=2))
