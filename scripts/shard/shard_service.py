"""Command line of the sharded exact-unlearning component (src/fu/shard_exact.py), the FU step the Digital Twin calls.

  python scripts/shard/shard_service.py init --output STATE --shards 4 --seed 50042
  python scripts/shard/shard_service.py unlearn --original-run STATE --forget-request REQ.json --method shard_full --output STATE2
  python scripts/shard/shard_service.py verify --original-run STATE2

init     trains the S shard models (R_base rounds each) and writes the state directory.
unlearn  loads a state, applies one deletion request with the chosen policy (--method) and writes the new state directory.
verify   retrains every included shard from scratch and compares the weights bit for bit.
A state directory holds state.json, the shard models, rows.npz, and the cost records of the operation that produced it:
result.json (summary), round_costs.csv and client_costs.csv. Output directories are never overwritten.

Request file (JSON): {"client": 3} deletes the whole client; {"client": 3, "forget_row_ids": [...]} deletes those rows
(row_id values of the train split).

Defaults follow the v2 protocol: data connected_v3_auxfit_seed42, 20 clients, Dirichlet alpha 0.5, partition seed 42, shard
assignment seed 7, train config 'standard', >= 4 clients per shard, merge below 3. Policies read the validation split only.
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.fu.shard_exact import POLICIES, ForgetRequest, ShardConfig, ShardedUnlearner, assign_shards, _py

DATA_DEFAULT = 'data/processed/connected_v3_auxfit_seed42'


def write_costs(directory, summary, round_costs, client_costs, layout=None):
    out = Path(directory)
    pd.DataFrame(round_costs).to_csv(out/'round_costs.csv', index=False)
    pd.DataFrame(client_costs).to_csv(out/'client_costs.csv', index=False)
    if layout is not None:
        (out/'layout.json').write_text(json.dumps(layout, indent=1), encoding='utf-8')
    (out/'result.json').write_text(json.dumps(_py(summary), indent=1), encoding='utf-8')


def layout_of(directory):
    return json.loads((Path(directory)/'layout.json').read_text(encoding='utf-8'))


def init_state(data, output, shards, seed, policy, clients=20, alpha=.5, partition_seed=42, assign_seed=7,
               train_config='standard', min_clients=4, merge_below=3):
    """Partition the train split, train the shard ensemble and save it. Returns (unlearner, fit result)."""
    if Path(output).exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    parts = partition(data['train']['y'], partition_seed, alpha, clients)
    config = ShardConfig.from_train_config(CONFIGS[train_config], min_clients_design=min_clients, merge_below=merge_below)
    fed = ShardedUnlearner(data['train'], data['validation'], parts, assign_shards(shards, clients, assign_seed), seed, policy, config)
    result = fed.fit(output)
    layout = {'clients': clients, 'alpha': alpha, 'partition_seed': partition_seed, 'assign_seed': assign_seed,
              'shards': shards, 'train_config': train_config}
    write_costs(output, {'operation': 'init', 'reference': result['reference'], 'cost': result['cost'],
                         'clients': fed.client_state()}, result['round_costs'], result['client_costs'], layout)
    return fed, result


def load_state(data, directory):
    layout = layout_of(directory)
    parts = partition(data['train']['y'], layout['partition_seed'], layout['alpha'], layout['clients'])
    return ShardedUnlearner.load(directory, data['train'], data['validation'], parts), layout


def read_request(data, path):
    spec = json.loads(Path(path).read_text(encoding='utf-8'))
    if set(spec) - {'client', 'forget_row_ids', 'kind'} or 'client' not in spec:
        raise ValueError('request needs "client" and optionally "forget_row_ids"')
    if 'forget_row_ids' not in spec:
        if spec.get('kind', 'client') != 'client':
            raise ValueError('a sample request needs "forget_row_ids"')
        return ForgetRequest(int(spec['client']))
    ids = np.asarray(spec['forget_row_ids']); train_ids = data['train']['row_id']
    order = np.argsort(train_ids, kind='stable'); pos = np.searchsorted(train_ids[order], ids)
    if not len(ids) or (pos >= len(train_ids)).any() or (train_ids[order][np.minimum(pos, len(train_ids) - 1)] != ids).any():
        raise ValueError('forget_row_ids must be row_id values of the train split')
    return ForgetRequest(int(spec['client']), order[pos])


def apply_request(data, original, request, output, policy=None):
    """Load `original`, apply one request, save to `output`. Returns (unlearner, UnlearningResult)."""
    if Path(output).exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    fed, layout = load_state(data, original)
    result = fed.unlearn(request, policy, output)
    summary = {k: v for k, v in result.to_json().items() if k not in ('round_costs', 'client_costs')}
    write_costs(output, {'operation': 'unlearn', **summary, 'clients': fed.client_state()}, result.round_costs,
                result.client_costs, layout)
    return fed, result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--data', type=Path, default=ROOT/DATA_DEFAULT)
    parser.add_argument('--threads', type=int, default=2)
    sub = parser.add_subparsers(dest='command', required=True)
    p = sub.add_parser('init')
    p.add_argument('--output', type=Path, required=True); p.add_argument('--shards', type=int, required=True)
    p.add_argument('--seed', type=int, required=True); p.add_argument('--method', choices=POLICIES, default='shard_full')
    p.add_argument('--clients', type=int, default=20); p.add_argument('--alpha', type=float, default=.5)
    p.add_argument('--partition-seed', type=int, default=42); p.add_argument('--assign-seed', type=int, default=7)
    p.add_argument('--train-config', choices=sorted(CONFIGS), default='standard')
    p.add_argument('--min-clients', type=int, default=4); p.add_argument('--merge-below', type=int, default=3)
    p = sub.add_parser('unlearn')
    p.add_argument('--original-run', type=Path, required=True); p.add_argument('--forget-request', type=Path, required=True)
    p.add_argument('--method', choices=POLICIES, default=None, help='policy; default: the one stored in the state')
    p.add_argument('--output', type=Path, required=True)
    p = sub.add_parser('verify')
    p.add_argument('--original-run', type=Path, required=True)
    args = parser.parse_args(argv)

    from src.data.loading import load_data
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    data = load_data(args.data)
    if args.command == 'init':
        fed, result = init_state(data, args.output, args.shards, args.seed, args.method, args.clients, args.alpha,
                                 args.partition_seed, args.assign_seed, args.train_config, args.min_clients, args.merge_below)
        print(json.dumps(_py({'output': args.output, 'reference': result['reference'],
                              'rounds': result['cost']['rounds'], 'seconds': result['cost']['wall_seconds_total']})))
    elif args.command == 'unlearn':
        fed, result = apply_request(data, args.original_run, read_request(data, args.forget_request), args.output, args.method)
        print(json.dumps(_py({'output': args.output, 'step': result.step, 'policy': result.policy, 'kind': result.kind,
                              'event': result.event, 'exact': result.exact, 'structure_ok': result.structure_ok,
                              'rounds': result.cost['rounds'], 'clients_selected': result.cost['clients_selected'],
                              'warnings': [w['code'] for w in result.structured_warnings]})))
    else:
        fed, _ = load_state(data, args.original_run)
        check = fed.verify_bit_exact()
        print(json.dumps(_py(check)))
        if not (check['bit_exact'] and check['exact'] and check['structure_ok'] and check['included_equals_active']):
            sys.exit(1)


if __name__ == '__main__':
    main()
