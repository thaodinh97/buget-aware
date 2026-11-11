"""Same-machine reference for the component: the unmodified v2 stream runner on a subset of the v2 training seeds.

  python scripts/shard/run_v2_stream_subset.py OUT --seeds 50042 50043 50044

Shard models trained on different machines differ in the last bits, and policies that stop on a threshold (utility_stop,
dynamicfu) can then take different round counts. So a replay through src/fu/shard_exact.py can only be compared bit for bit
with a v2 run made on the same machine. This script registers a design 'v2_subset' in memory (the v2 design with fewer
seeds) and executes scripts/shard/run_shard_stream_v2.py as it is; shard_v2.start_run still checks the data content hash and
writes OUT/run.json. Rows of one seed do not depend on the other seeds, so they equal the rows a full v2 run would produce
on this machine. The output is a reference for scripts/shard/replay_v2_stream.py --compare, never evidence for the v2 gates
(scripts/shard/evaluate_shard_v2.py rejects it: the design is not 'v2').
"""
import argparse
import os
import runpy
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import scripts.shard.shard_v2 as v2


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seeds', type=int, nargs='+', required=True)
    args = parser.parse_args(argv)
    unknown = sorted(set(args.seeds) - set(v2.DESIGNS['v2']['seeds']))
    if unknown or len(set(args.seeds)) != len(args.seeds):
        raise ValueError(f'seeds must be distinct v2 seeds {v2.DESIGNS["v2"]["seeds"]}; got {args.seeds}')
    v2.DESIGNS['v2_subset'] = {**v2.DESIGNS['v2'], 'seeds': list(args.seeds)}
    os.environ['SHARD_DESIGN'] = 'v2_subset'
    sys.argv = [str(ROOT/'scripts'/'shard'/'run_shard_stream_v2.py'), str(args.output)]
    runpy.run_path(sys.argv[0], run_name='__main__')


if __name__ == '__main__':
    main()
