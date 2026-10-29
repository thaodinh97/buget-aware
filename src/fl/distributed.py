"""Safe NumPy transport helpers for the Docker-based FedAvg prototype."""
from io import BytesIO

import numpy as np
import torch

from src.fl.fedavg_core import aggregate


def pack_state(state, round_id):
    arrays = {'__round__': np.asarray(round_id, dtype=np.int64)}
    arrays.update({f'tensor::{name}': value.detach().cpu().numpy() for name, value in state.items()})
    stream = BytesIO()
    np.savez_compressed(stream, **arrays)
    return stream.getvalue()


def unpack_state(payload, expected_state, expected_round):
    try:
        with np.load(BytesIO(payload), allow_pickle=False) as archive:
            if '__round__' not in archive.files or int(archive['__round__']) != expected_round:
                raise ValueError('Unexpected model round')
            expected_keys = {f'tensor::{key}' for key in expected_state}
            actual_keys = set(archive.files) - {'__round__'}
            if actual_keys != expected_keys:
                raise ValueError('Unexpected model state keys')
            result = {}
            for key, expected in expected_state.items():
                array = archive[f'tensor::{key}']
                if array.shape != tuple(expected.shape) or array.dtype != expected.numpy().dtype:
                    raise ValueError(f'Invalid model tensor: {key}')
                if np.issubdtype(array.dtype, np.floating) and not np.isfinite(array).all():
                    raise ValueError(f'Nonfinite model tensor: {key}')
                result[key] = torch.from_numpy(array.copy())
            return result
    except (OSError, KeyError, TypeError) as exc:
        raise ValueError('Invalid model payload') from exc


def aggregate_updates(updates):
    if not updates:
        raise ValueError('No client updates to aggregate')
    return aggregate([state for state, _ in updates], [count for _, count in updates])
