"""Verify one seed run of scripts/run_dynamicfu_sample.py: artifact hashes, recorded checks, checkpoint
hashes against the recorded trajectories, and bit-for-bit re-execution of one original shard and of the
first natural and first canary request of the first alpha (canary original, trajectory, both arms,
contribution score) with their test metrics recomputed."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scripts.experiment_context import use_run_device
from scripts.run_dynamicfu_sample import (alpha_tag, verify_hashes, load_inputs, stored_requests, client_workers,
                                          scorer_for, Request, eval_indices, request_seed, trajectory_signals)
from src.data.prepare import digest
from src.data.canary import stamp_combination
from src.eval.metrics import utility
from src.fl.fedavg_core import ClientWorker, state
from src.fl.sharded import init_shard, shard_workers, train_shard, train_sharded, margins, ensemble_margin, state_hash
from src.fu.shard_retrain import retrain_trajectory
from src.fu.dynamicfu import ControllerConfig, initial_eta, make_signal_fn, controlled_retrain


def ckpt(run, name):
    return torch.load(run / 'checkpoints' / f'{name}.pt', weights_only=True)['state_dict']


def verify(run, data_path):
    use_run_device(run)
    verify_hashes(run)
    manifest = json.loads((run / 'manifest.json').read_text())
    status = json.loads((run / 'status.json').read_text())
    assert status['all_checks_passed'] and status['failed'] == 0, status
    checks = json.loads((run / 'checks.json').read_text())
    assert all(all(c.values()) for c in checks.values())
    req_dir = Path(manifest['requests'])
    verify_hashes(req_dir)
    assert digest(req_dir / 'manifest.json') == manifest['requests_manifest_sha256']
    assert digest(data_path / 'manifest.json') == manifest['data_manifest_sha256']
    cfg, seed = manifest['config'], manifest['seed']
    ccfg = ControllerConfig(**manifest['controller'])
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    data, train, _ = load_inputs(data_path, manifest['smoke_only'])
    val, test = data['validation'], data['test']
    fl, rounds = cfg['fl'], cfg['fl']['rounds']
    dim, hidden = train['X'].shape[1], cfg['hidden']
    init_fn = lambda s: init_shard(dim, hidden, seed, s)
    hashes = json.loads((run / 'state_hashes.json').read_text())
    metrics = pd.read_csv(run / 'metrics.csv')
    trigger = json.loads((req_dir / 'manifest.json').read_text())['trigger']
    counts = {'requests': 0, 'checkpoints': 0, 'reexecuted': 0}
    for key, h in hashes.items():
        tag, rid = key.split('/')
        row = metrics[(metrics.alpha.map(alpha_tag) == tag) & (metrics.request == rid)].set_index('method')
        s_u = int(row.loc['dynamicfu', 'shard'])
        assert h['trajectory'][-1] == h['full_retrain'][s_u]
        for name in ['dynamicfu', 'dynamicfu_ffree']:
            assert state_hash(ckpt(run, f'{tag}_{rid}_{name}_shard{s_u}')) == h[name] == h['trajectory'][int(row.loc[name, 'rounds'])]
        assert state_hash(ckpt(run, f'{tag}_{rid}_affected_shard_retrain_shard{s_u}')) == h['trajectory'][-1]
        counts['checkpoints'] += 3
        counts['requests'] += 1
    alpha = cfg['alphas'][0]
    tag = alpha_tag(alpha)
    _, record, parts, shards, arrays = stored_requests(req_dir, cfg, train, alpha)
    P = json.loads((req_dir / 'manifest.json').read_text())['partition_seed']
    workers = client_workers(train, parts)
    natural = [init_fn(s) for s in range(len(shards))]
    for s, m in enumerate(natural):
        m.load_state_dict(ckpt(run, f'{tag}_original_shard{s}'))
    small = int(np.argmin([x['rows'] for x in record['shards']]))
    again, _ = train_shard(dim, hidden, shard_workers(shards[small], workers), fl, rounds, seed, small)
    assert state_hash(state(again)) == state_hash(state(natural[small])), 'original shard not reproduced'
    counts['reexecuted'] += 1
    natural_scorer = scorer_for(cfg, natural, init_fn, seed, train['y'], val, workers, shards)
    backdoor_x = stamp_combination(test['X'][test['y'] == 1], trigger)
    for kind in ['natural', 'canary']:
        spec = next((s for s in record['requests'] if s['kind'] == kind), None)
        if spec is None:
            continue
        key = f'{tag}/{spec["id"]}'
        req = Request(spec, tag, train, parts, shards, workers, arrays, trigger, cfg, P)
        s_u, members = req.shard, shards[req.shard]
        original, scorer = natural, natural_scorer
        if kind == 'canary':
            cx, cy, cids = req.forget
            c = req.client
            aug = dict(workers)
            aug[c] = ClientWorker(c, np.r_[train['X'][parts[c]], cx], np.r_[train['y'][parts[c]], cy],
                                  np.r_[train['row_id'][parts[c]], cids])
            shard_model, _ = train_shard(dim, hidden, shard_workers(members, aug), fl, rounds, seed, s_u)
            assert state_hash(state(shard_model)) == state_hash(ckpt(run, f'{tag}_{req.id}_original_shard{s_u}'))
            original = list(natural)
            original[s_u] = shard_model
            scorer = scorer_for(cfg, original, init_fn, seed, np.r_[train['y'], cy], val, aug, shards)
        assert [state_hash(state(m)) for m in original] == hashes[key]['original']
        ev = eval_indices(parts, members, req.client, req.retain, P, s_u, cfg['controller']['eval_rows'])
        signal = make_signal_fn(original, s_u, (train['X'][ev], train['y'][ev]), (val['X'], val['y']))
        states, _ = retrain_trajectory(init_fn(s_u), shard_workers(members, req.retained), fl, rounds, seed)
        assert [state_hash(x) for x in states] == hashes[key]['trajectory'], f'{key}: trajectory not reproduced'
        score = scorer.score(scorer.raw(s_u, ClientWorker(req.client, *req.forget), request_seed(seed, req.client, req.fraction)))
        rows = metrics[(metrics.alpha == alpha) & (metrics.request == req.id)].set_index('method')
        np.testing.assert_allclose(score['I'], rows.loc['dynamicfu', 'I_F'], rtol=1e-12, atol=0)
        m, logs, trace = controlled_retrain(init_fn(s_u), shard_workers(members, req.retained), fl, seed, ccfg,
                                            initial_eta(score['I'], ccfg), scorer.f1_original, signal)
        assert state_hash(state(m)) == hashes[key]['dynamicfu'] and len(logs) == int(rows.loc['dynamicfu', 'rounds'])
        assert trace == json.loads((run / 'traces' / f'{tag}_{req.id}_dynamicfu.json').read_text())
        zs = [margins(x, test['X']) for x in original]
        zs[s_u] = margins(m, test['X'])
        u = utility(test['y'], expit(ensemble_margin(zs)))
        np.testing.assert_allclose([u['macro_f1'], u['fpr']], [rows.loc['dynamicfu', 'macro_f1'], rows.loc['dynamicfu', 'fpr']],
                                   rtol=1e-12, atol=0)
        zb = [margins(x, backdoor_x) for x in original]
        zb[s_u] = margins(m, backdoor_x)
        np.testing.assert_allclose(np.mean(1 - expit(ensemble_margin(zb))), rows.loc['dynamicfu', 'backdoor_p_normal'], rtol=1e-12)
        counts['reexecuted'] += 3 + (kind == 'canary')
    print(json.dumps({'verified': True, **counts}))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path('data/processed/connected_v3_seed42'))
    args = parser.parse_args()
    verify(args.run, args.data)
