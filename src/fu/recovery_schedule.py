"""Interleaved forget-recovery: stage -> 1 FedAvg round -> stage -> 1 round ... (`rounds` rounds in total).

Round r is seeded with seed + r*10000, which is exactly the per-round seed of
fedavg(start, workers, fl, rounds, seed), so a no-op stage reproduces standard
recovery. F enters only the local stages, never a FedAvg update, and the last
step is always a recovery round.
"""
from src.fl.fedavg_core import fedavg


def interleaved(start, stage, workers, fl, rounds, seed):
    """stage(model) -> (model, info). Returns the final model and the per-stage infos."""
    m, infos = start, []
    for r in range(rounds):
        m, info = stage(m); infos.append(info)
        m, _ = fedavg(m, workers, fl, 1, seed + r*10000)
    return m, infos
