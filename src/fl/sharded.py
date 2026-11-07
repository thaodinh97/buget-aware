"""Sharded FedAvg: disjoint client shards, one FedAvg model per shard, logit-sum ensemble.

Follows the shard architecture of Wu et al., "DynamicFU" (Sensors 26:3714, 2026), Sec. 3.1, Eq. (1)-(5).
Every shard is trained by the unchanged `fedavg` core so that retraining a shard on the same clients
and seeds reproduces it bit for bit; exact unlearning relies on that.
"""
import hashlib
import numpy as np
import torch
from scipy.special import expit
from src.fl.fedavg_core import model, fedavg


def assign_shards(n_clients, n_shards, seed):
    """Seeded random grouping of client ids into `n_shards` disjoint, near-equal-size shards."""
    if n_shards < 1 or n_clients < n_shards:
        raise ValueError('Need at least one client per shard')
    order = np.random.default_rng(seed).permutation(n_clients)
    return [np.sort(part).astype(np.int64) for part in np.array_split(order, n_shards)]


def shard_of_client(shards):
    return {int(c): s for s, members in enumerate(shards) for c in members}


def shard_init_seed(seed, shard):
    return seed + 1000 * (shard + 1)


def init_shard(input_dim, hidden, seed, shard):
    """Fresh model with the shard's fixed initialisation; used for training and every retrain."""
    return model(input_dim, hidden, shard_init_seed(seed, shard))


def shard_workers(members, workers_by_client):
    """Workers of a shard in ascending client-id order. FedAvg sums updates in list order, so original
    training, full retraining and shard retraining must all build their lists here."""
    return [workers_by_client[int(c)] for c in members if int(c) in workers_by_client]


def train_shard(input_dim, hidden, workers, config, rounds, seed, shard):
    if not workers:
        raise ValueError(f'Shard {shard} has no participating client')
    return fedavg(init_shard(input_dim, hidden, seed, shard), workers, config, rounds, seed)


def train_sharded(input_dim, hidden, shards, workers_by_client, config, rounds, seed):
    models, logs = [], []
    for s, members in enumerate(shards):
        m, log = train_shard(input_dim, hidden, shard_workers(members, workers_by_client), config, rounds, seed, s)
        models.append(m)
        logs.append(log)
    return models, logs


def fedavg_round(current, workers, config, round_id, seed):
    """Round `round_id` of `fedavg(..., seed)` run in isolation: identical worker seeds, so looping over
    rounds reproduces one multi-round call bit for bit while allowing an online stopping rule."""
    result, logs = fedavg(current, workers, config, 1, seed + round_id * 10000)
    logs[0]['round'] = round_id
    return result, logs[0]


def margins(m, x, batch_size=4096):
    """Binary logit l1 - l0 per row; softmax(l)[1] == sigmoid(l1 - l0)."""
    m.eval()
    device = next(m.parameters()).device
    out = []
    with torch.no_grad():
        for i in range(0, len(x), batch_size):
            logits = m(torch.as_tensor(x[i:i + batch_size], dtype=torch.float32, device=device))
            out.append((logits[:, 1] - logits[:, 0]).double().cpu().numpy())
    return np.concatenate(out) if out else np.zeros(0)


def ensemble_margin(margin_list):
    """Eq. (5): z(x) = sum_s f(x; w_s), summed in shard order."""
    total = np.zeros_like(margin_list[0])
    for z in margin_list:
        total = total + z
    return total


def ensemble_probability(margin_list):
    return expit(ensemble_margin(margin_list))


def state_hash(state_dict):
    digest = hashlib.sha256()
    for key in sorted(state_dict):
        tensor = state_dict[key].detach().cpu().contiguous()
        digest.update(key.encode())
        digest.update(str(tensor.dtype).encode())
        digest.update(tensor.numpy().tobytes())
    return digest.hexdigest()


def states_equal(a, b):
    return a.keys() == b.keys() and all(torch.equal(a[k].cpu(), b[k].cpu()) for k in a)


def param_l2(states_a, states_b):
    """L2 distance between two shard ensembles, all shard parameters concatenated."""
    if len(states_a) != len(states_b):
        raise ValueError('Ensembles differ in shard count')
    total = 0.
    for a, b in zip(states_a, states_b):
        total += sum(float(((a[k].double() - b[k].double()) ** 2).sum()) for k in a)
    return total ** .5
