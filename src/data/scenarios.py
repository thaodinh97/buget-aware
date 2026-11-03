"""Deletion scenarios in which the influence of F on the model can be identified."""
import numpy as np
from src.data.federated import partition


def exclusive_family_partition(y, types, family, target, seed, alpha=.5, clients=5):
    """Synthetic ownership: every training row of `family` belongs to the target client.

    Other rows follow the usual label Dirichlet. Retraining without the family then never sees
    it, unlike random deletion where other clients keep equivalent records.
    """
    types = np.asarray(types)
    family_rows = np.flatnonzero(types == family)
    if family == 'normal' or not len(family_rows) or target not in range(clients):
        raise ValueError('Family must be a present attack family and target a valid client')
    rest = np.flatnonzero(types != family)
    result = [rest[part] for part in partition(y[rest], seed, alpha, clients)]
    result[target] = np.sort(np.concatenate([result[target], family_rows]))
    return result


def family_request(types, client_indices, family, fraction, seed):
    """Forget a fraction (1.0 = all) of the target's rows of one family."""
    if not 0 < fraction <= 1:
        raise ValueError('Fraction must be in (0, 1]')
    rows = client_indices[np.asarray(types)[client_indices] == family]
    count = len(rows) if fraction == 1 else int(np.floor(len(rows) * fraction))
    if count < 1 or count >= len(client_indices):
        raise ValueError('Family request is empty or removes the whole client')
    f = np.sort(np.random.default_rng(seed).permutation(rows)[:count]).astype(np.int64)
    return f, np.setdiff1d(client_indices, f)


def unique_feature_rows(x):
    """True for rows whose encoded vector occurs exactly once in x."""
    _, inverse, counts = np.unique(np.ascontiguousarray(x), axis=0, return_inverse=True, return_counts=True)
    return counts[inverse.ravel()] == 1


def atypical_request(y, losses, client_indices, fraction, unique_mask):
    """Per class, forget the highest-loss rows of the original model that have no exact duplicate.

    Chosen from the original model and metadata only, never from FU results or test data.
    """
    forgotten = []
    for label in [0, 1]:
        rows = client_indices[y[client_indices] == label]
        count = int(np.floor(len(rows) * fraction))
        candidates = rows[unique_mask[rows]]
        if count < 1 or count >= len(rows) or count > len(candidates):
            raise ValueError('Atypical request cannot be filled for every class')
        order = np.lexsort((candidates, -losses[candidates]))
        forgotten.extend(candidates[order[:count]])
    f = np.array(sorted(forgotten), dtype=np.int64)
    return f, np.setdiff1d(client_indices, f)
