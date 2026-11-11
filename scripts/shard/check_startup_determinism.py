"""Is the first training of a process bit-reproducible on this machine? Usage:

  python scripts/shard/check_startup_determinism.py --runs 200 [--parallel 4] [--mode plain warm] [--threads 2]

Starts `--runs` fresh processes per mode. Each one trains client 0 of the v2 partition for one local round from the same
seeded model (src.fl.fedavg_core only, 'standard' train config) as the very first training of the process, and prints a
hash of the resulting weights. One distinct hash per mode means the first training is reproducible; more than one means
that a model trained first in a process can differ in the last bits from a retrain of it, so bit-for-bit checks
(shard_exact.verify_bit_exact, G3 of the v2 protocol) can fail without any deletion being involved.
  plain  nothing runs before the measured training (the situation of the v2 runners)
  warm   src.fu.shard_exact.warm_up runs first (the situation of the component)
Rare events need many runs: with a rate of 1 in 100, 200 runs still miss it in about one case out of seven.
"""
import argparse
import collections
import hashlib
import os
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT/'data'/'processed'/'connected_v3_auxfit_seed42'/'train.npz'


def child(mode, threads):
    sys.path.insert(0, str(ROOT))
    import numpy as np
    import torch
    from scripts.shard.train_config import CONFIGS
    from src.data.federated import partition
    from src.fl.fedavg_core import ClientWorker, model
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)
    with np.load(DATA, allow_pickle=False) as arrays:
        X, y, row_id = arrays['X'], arrays['y'], arrays['row_id']
    rows = partition(y, 42, .5, 20)[0]
    fl = CONFIGS['standard']['fl']
    if mode == 'warm':
        from src.fu.shard_exact import ShardConfig, warm_up
        warm_up(X.shape[1], ShardConfig.from_train_config(CONFIGS['standard']))
    worker = ClientWorker(0, X[rows], y[rows], row_id[rows], 'cpu')
    update, _, log = worker.train(model(X.shape[1], [64, 32], 50044, 'cpu'), fl, 50044)
    digest = hashlib.sha256(b''.join(v.numpy().tobytes() for v in update.values())).hexdigest()[:12]
    print(f'{digest} loss={log["loss"]!r} first_training_seconds={log["compute_seconds"]:.3f}')


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--runs', type=int, default=200)
    parser.add_argument('--parallel', type=int, default=1)
    parser.add_argument('--mode', nargs='+', choices=['plain', 'warm'], default=['plain', 'warm'])
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--child', choices=['plain', 'warm'], help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.child:
        return child(args.child, args.threads)
    env = {**os.environ, 'FL_DEVICE': 'cpu'}

    def run(mode):
        done = subprocess.run([sys.executable, __file__, '--child', mode, '--threads', str(args.threads)], env=env,
                              capture_output=True, text=True)
        return mode, (done.stdout.strip().splitlines() or [f'FAILED {done.stderr.strip()[-200:]}'])[-1]

    jobs = [mode for _ in range(args.runs) for mode in args.mode]          # interleaved, so both modes see the same load
    with ThreadPoolExecutor(args.parallel) as pool:
        outcomes = list(pool.map(run, jobs))
    for mode in args.mode:
        lines = [line for m, line in outcomes if m == mode]
        counts = collections.Counter(line.split(' first_training_seconds')[0] for line in lines)
        seconds = sorted(float(line.rsplit('=', 1)[1]) for line in lines if 'first_training_seconds=' in line)
        print(f'{mode}: {len(lines)} processes, {len(counts)} distinct result(s); '
              f'median seconds of the first training {seconds[len(seconds)//2] if seconds else float("nan"):.3f}')
        for value, count in counts.most_common():
            print(f'  {count:5d}  {value}')


if __name__ == '__main__':
    main()
