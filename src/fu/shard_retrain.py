"""Retraining of the shard that holds a forget set, in sharded FedAvg (Wu et al. 2026, Sec. 4.4).

The affected shard restarts from its original initialisation and trains on retained data only; the
other shards are reused untouched. Round r runs with the seeds of round r of `fedavg(..., rounds, seed)`,
so every fixed budget is a prefix of one trajectory and the full budget equals retraining every shard.
"""
from src.fl.fedavg_core import state
from src.fl.sharded import fedavg_round


def affected_shards(shards, owners):
    """Shards that contain any client owning a forgotten row."""
    owners = {int(c) for c in owners}
    found = sorted(s for s, members in enumerate(shards) if owners & set(map(int, members)))
    if not found or sum(len(owners & set(map(int, shards[s]))) for s in found) != len(owners):
        raise ValueError('Forget owners are not all assigned to a shard')
    return found


def retrain_trajectory(init_model, workers, config, rounds, seed, on_round=None):
    """states[r] = affected shard after r rounds (states[0] = initialisation); logs[r-1] = FedAvg log.
    `on_round(r, model)` observes each round without changing it."""
    if not workers:
        raise ValueError('The affected shard has no client left after the request')
    current, states, logs = init_model, [state(init_model)], []
    if on_round:
        on_round(0, current)
    for r in range(rounds):
        current, log = fedavg_round(current, workers, config, r, seed)
        states.append(state(current))
        logs.append(log)
        if on_round:
            on_round(r + 1, current)
    return states, logs
