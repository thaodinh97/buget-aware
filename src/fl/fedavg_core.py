"""Shared MLP/local optimizer/FedAvg core for original, retrain and recovery."""
import copy
import os
import time
import numpy as np
import torch
from torch import nn


def get_device(device=None):
    requested = device or os.environ.get('FL_DEVICE', 'auto')
    if requested != 'cpu':
        os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')
    if requested == 'auto':
        return torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    resolved = torch.device(requested)
    if resolved.type == 'cuda' and not torch.cuda.is_available():
        raise RuntimeError(f'CUDA device requested ({resolved}) but CUDA is unavailable')
    return resolved


def model(input_dim, hidden, seed, device=None):
    torch.manual_seed(seed)
    return nn.Sequential(nn.Linear(input_dim, hidden[0]), nn.ReLU(),
                         nn.Linear(hidden[0], hidden[1]), nn.ReLU(), nn.Linear(hidden[1], 2)).to(get_device(device))


def state(model):
    return {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}


def aggregate(updates, counts):
    if not updates or len(updates) != len(counts) or any(n <= 0 for n in counts):
        raise ValueError('Invalid FedAvg participants/counts')
    total = sum(counts)
    return {k: sum(u[k] * (n / total) for u, n in zip(updates, counts)) for k in updates[0]}


def batches(n, batch_size, seed):
    if n <= 0 or batch_size <= 0:
        raise ValueError('Empty data or invalid batch size')
    order = np.random.default_rng(seed).permutation(n)
    for start in range(0, n, batch_size):
        yield order[start:start + batch_size]


class ClientWorker:
    """Raw/transformed records stay local to this logical worker."""
    def __init__(self, client_id, x, y, row_ids, device=None):
        if len(y) == 0 or len(x) != len(y) or len(row_ids) != len(y):
            raise ValueError('Invalid client data')
        self.client_id = client_id
        self.device = get_device(device)
        self.x = torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.y = torch.as_tensor(y, dtype=torch.long, device=self.device)
        self.row_ids = np.array(row_ids)

    def train(self, global_model, config, seed):
        start = time.perf_counter()
        local = copy.deepcopy(global_model)
        optimizer = torch.optim.Adam(local.parameters(), lr=config['lr'])
        steps, loss_sum = 0, 0.
        local.train()
        for epoch in range(config['local_epochs']):
            for idx in batches(len(self.y), config['batch_size'], seed + epoch):
                optimizer.zero_grad()
                loss = nn.functional.cross_entropy(local(self.x[idx]), self.y[idx])
                if not torch.isfinite(loss):
                    raise FloatingPointError('Nonfinite training loss')
                loss.backward()
                optimizer.step()
                loss_sum += loss.item()
                steps += 1
        return state(local), len(self.y), {'client_id': self.client_id, 'steps': steps,
                'loss': loss_sum / steps, 'compute_seconds': time.perf_counter() - start}


def fedavg(initial, workers, config, rounds, seed, observer=None):
    result = copy.deepcopy(initial)
    logs = []
    size = sum(v.numel() * v.element_size() for v in result.state_dict().values())
    for round_id in range(rounds):
        before = state(result) if observer is not None else None
        started = time.perf_counter()
        updates, counts, local_logs = [], [], []
        for worker in workers:
            update, count, log = worker.train(result, config, seed + round_id * 10000 + worker.client_id * 100)
            updates.append(update)
            counts.append(count)
            local_logs.append(log)
        result.load_state_dict(aggregate(updates, counts))
        logs.append({'round': round_id, 'clients': local_logs, 'counts': counts,
                     'weights': [n / sum(counts) for n in counts],
                     'wall_seconds': time.perf_counter() - started,
                     'payload_bytes_up': size * len(workers), 'payload_bytes_down': size * len(workers),
                     'transport': 'logical_sequential_simulation_not_wire_measurement'})
        if observer is not None:
            observer(round_id, before, state(result), updates, counts, logs[-1])
    return result, logs


def predict(model, x, batch_size=4096):
    model.eval()
    device = next(model.parameters()).device
    with torch.no_grad():
        return np.concatenate([torch.softmax(model(torch.as_tensor(
            x[i:i+batch_size], dtype=torch.float32, device=device)), 1)[:, 1].cpu().numpy()
                               for i in range(0, len(x), batch_size)])
