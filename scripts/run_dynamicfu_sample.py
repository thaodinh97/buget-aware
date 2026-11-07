"""DynamicFU adapted to sample-level deletion on sharded FedAvg (ToN-IoT; post-freeze FU track).

Protocol and gates: docs/dynamicfu_sample_plan.md. Stages:
  requests  one partition: train the first-seed originals, score every candidate stratified request
            with the contribution I_F, select `per_group` requests from each contribution tertile, and
            build combination-canary requests on the clients with the most attack rows;
  tune      on the pilot partition: record the retrain trajectory signals of its natural requests and
            choose (rho, mu, tau_ctrl, tau_drop) by the predeclared validation-only rule;
  run       one training seed on the main partition: originals, full retrain, the affected-shard
            trajectory (every fixed budget), DynamicFU and the F-free ablation, with the gate-1 checks.
"""
import argparse
import itertools
import json
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.run_pilot import load_data
from src.data.prepare import digest, write_json
from src.data.federated import partition_min_size, forget_request, ids_hash
from src.data.canary import combination_trigger, make_combination_canaries, stamp_combination
from src.fl.fedavg_core import ClientWorker, model, state, fedavg, get_device
from src.fl.sharded import (assign_shards, shard_of_client, init_shard, shard_workers, train_sharded, train_shard,
                            margins, ensemble_margin, state_hash, states_equal)
from src.fu.shard_retrain import affected_shards, retrain_trajectory
from src.fu.dynamicfu import (ContributionConfig, ContributionScorer, ControllerConfig, initial_eta, replay,
                              make_signal_fn, controlled_retrain, binary_distribution)
from src.eval.metrics import utility, similarity, calibrate_mia, matched_indices, mia
from scipy.special import expit

SPLITS = ['train', 'validation', 'calibration', 'test']
METHODS = ['original', 'full_retrain', 'affected_shard_retrain', 'rapid_retrain', 'dynamicfu', 'dynamicfu_ffree',
           'global_original']


# ---------------------------------------------------------------- shared setup
def alpha_tag(alpha):
    return f'a{alpha:g}'


def smoke_config(cfg):
    cfg = json.loads(json.dumps(cfg))
    middle = len(cfg['alphas']) // 2
    cfg['alphas'] = cfg['alphas'][middle:middle + 1]
    cfg['fl']['rounds'] = 3
    cfg['controller'].update(rounds_max=3, eval_rows=200)
    cfg['rapid_rounds'] = 2
    cfg['seeds'] = cfg['seeds'][:3]
    cfg['requests'].update(per_group=1, canary_count=50, canary_min_attack_rows=100, max_canary_clients=1)
    cfg['tuning']['grid'] = {'rho': [.5], 'mu': [1.], 'tau_ctrl': [.01], 'tau_drop': [.05, 1.]}
    cfg['tuning'].update(max_mean_val_f1_loss=1., max_val_f1_loss=1.)
    return cfg


def load_inputs(data_path, smoke):
    data = load_data(data_path)
    names = json.loads((data_path / 'transformer.json').read_text())['feature_names']
    train = data['train']
    if smoke:  # explicit smoke-only stratified cap, never experimental evidence
        idx = np.concatenate([np.flatnonzero(train['y'] == c)[:1000] for c in [0, 1]])
        train = {k: v[idx] for k, v in train.items()}
    return data, train, names


def federation(y, cfg, alpha, partition_seed):
    parts, attempts = partition_min_size(y, partition_seed, alpha, cfg['clients'], cfg['min_client_rows'],
                                         cfg['max_partition_attempts'])
    return parts, attempts, assign_shards(cfg['clients'], cfg['shards'], partition_seed)


def client_workers(train, parts):
    return {c: ClientWorker(c, train['X'][p], train['y'][p], train['row_id'][p]) for c, p in enumerate(parts)}


def contribution_cfg(cfg):
    return ContributionConfig(**cfg['contribution'])


def controller_cfg(cfg, selected):
    c = cfg['controller']
    return ControllerConfig(rounds_base=cfg['fl']['rounds'], rounds_max=c['rounds_max'], eta_min=c['eta_min'],
                            eta_max=c['eta_max'], omega_g=c['omega_g'], omega_s=c['omega_s'], **selected)


def scorer_for(cfg, models, init_fn, seed, train_labels, val, workers, shards):
    scorer = ContributionScorer(contribution_cfg(cfg), models, init_fn, cfg['fl'], seed,
                                binary_distribution(train_labels), (val['X'], val['y']))
    scorer.fit_pool(workers, shard_of_client(shards))
    return scorer


def request_seed(seed, client, fraction):
    return seed + 9000 + 100 * client + (99 if fraction is None else int(round(fraction * 100)))


def eval_indices(parts, members, target, retain, partition_seed, shard, rows):
    """The controller's V set: a fixed subsample of the affected shard's retained training rows (F-free)."""
    idx = np.concatenate([retain if (retain is not None and int(k) == target) else parts[int(k)] for k in members])
    pick = np.random.default_rng(partition_seed + 13 + shard).permutation(len(idx))[:rows]
    return np.sort(idx[pick])


def trigger_for(data, train, names, cfg):
    return combination_trigger([data[s]['X'] for s in SPLITS], train['X'], names, cfg['requests']['combination_features'])


def canary_rows(train, parts, client, trigger, cfg, partition_seed):
    r = cfg['requests']
    cx, cy, src = make_combination_canaries(train['X'], train['y'], parts[client], r['canary_count'], trigger,
                                            partition_seed + r['canary_seed_offset'])
    return cx, cy, np.array([f'canary_c{client}_{i}' for i in range(len(cy))]), src


def verify_hashes(directory):
    for name, sha in json.loads((directory / 'artifact_hashes.json').read_text()).items():
        if digest(directory / name) != sha:
            raise ValueError(f'Changed artifact {directory / name}')


def finalize(output):
    write_json(output / 'artifact_hashes.json', {str(p.relative_to(output)).replace('\\', '/'): digest(p)
               for p in output.rglob('*') if p.is_file() and p.name != 'artifact_hashes.json'})


def base_manifest(cfg, data_path, smoke, device):
    return {'config': cfg, 'smoke_only': smoke, 'data_manifest_sha256': digest(data_path / 'manifest.json'),
            'device': str(device), 'python': platform.python_version(), 'torch': torch.__version__,
            'numpy': np.__version__, 'platform': platform.platform(), 'processor': platform.processor(),
            'sources': {str(p).replace('\\', '/'): digest(p) for root in ['src', 'scripts'] for p in Path(root).rglob('*.py')},
            'algorithm': 'dynamicfu_inspired (Wu et al. 2026), sample-level adaptation'}


def setup(args):
    cfg = json.loads(args.config.read_text())
    if args.smoke:
        cfg = smoke_config(cfg)
    torch.set_num_threads(cfg['threads'])
    torch.use_deterministic_algorithms(True)
    return cfg, get_device()


def stored_requests(req_dir, cfg, train, alpha):
    manifest = json.loads((req_dir / 'manifest.json').read_text())
    record = manifest['alphas'][alpha_tag(alpha)]
    parts, attempts, shards = federation(train['y'], cfg, alpha, manifest['partition_seed'])
    if attempts != record['partition_attempts'] or [ids_hash(train['row_id'][p]) for p in parts] != record['client_hashes']:
        raise ValueError('Regenerated partition differs from the request manifest')
    with np.load(req_dir / 'requests.npz') as z:
        arrays = {k: z[k] for k in z.files}
    return manifest, record, parts, shards, arrays


# ---------------------------------------------------------------- stage: requests
def stage_requests(args):
    cfg, device = setup(args)
    out = args.output
    if out.exists():
        raise FileExistsError(f'Refusing overwrite: {out}')
    data, train, names = load_inputs(args.data, args.smoke)
    val, P, seed0, fl, r = data['validation'], args.partition_seed, cfg['seeds'][0], cfg['fl'], cfg['requests']
    dim, hidden = train['X'].shape[1], cfg['hidden']
    init_fn = lambda s: init_shard(dim, hidden, seed0, s)
    trigger = trigger_for(data, train, names, cfg)
    out.mkdir(parents=True)
    manifest = base_manifest(cfg, args.data, args.smoke, device) | {
        'stage': 'requests', 'partition_seed': P, 'selection_training_seed': seed0, 'trigger': trigger, 'alphas': {}}
    arrays = {}
    started = time.perf_counter()
    for ai, alpha in enumerate(cfg['alphas']):
        tag = alpha_tag(alpha)
        parts, attempts, shards = federation(train['y'], cfg, alpha, P)
        owner = shard_of_client(shards)
        workers = client_workers(train, parts)
        original, _ = train_sharded(dim, hidden, shards, workers, fl, fl['rounds'], seed0)
        scorer = scorer_for(cfg, original, init_fn, seed0, train['y'], val, workers, shards)
        candidates, invalid = [], []
        for c in range(len(parts)):
            for fr in r['fractions']:
                rid = f'c{c}_f{fr:g}'
                try:
                    f, _ = forget_request(train['y'], parts[c], fr, P + r['request_seed_offset'])
                except ValueError as error:
                    invalid.append({'request': rid, 'reason': str(error)})
                    continue
                raw = scorer.raw(owner[c], ClientWorker(c, train['X'][f], train['y'][f], train['row_id'][f]),
                                 request_seed(seed0, c, fr))
                candidates.append({'id': rid, 'client': c, 'fraction': fr, 'shard': owner[c], 'forget_rows': int(len(f)),
                                   'forget_labels': np.bincount(train['y'][f], minlength=2).tolist(),
                                   'forget_hash': ids_hash(train['row_id'][f]), **raw, **scorer.score(raw)})
                arrays[f'{tag}__{rid}__forget'] = f
        order = sorted(candidates, key=lambda x: (-x['I'], x['id']))
        rng = np.random.default_rng(P + r['selection_seed_offset'] + ai)
        selected = []
        for group, members in zip(r['groups'], np.array_split(np.arange(len(order)), len(r['groups']))):
            for i in sorted(rng.choice(len(members), min(r['per_group'], len(members)), replace=False).tolist()):
                selected.append(order[members[i]] | {'group': group, 'kind': 'natural'})
        attack = [int((train['y'][p] == 1).sum()) for p in parts]
        eligible = [c for c in sorted(range(len(parts)), key=lambda c: (-attack[c], c)) if attack[c] >= r['canary_min_attack_rows']]
        canaries = []
        for c in eligible[:r['max_canary_clients']]:
            _, _, _, src = canary_rows(train, parts, c, trigger, cfg, P)
            arrays[f'{tag}__canary_c{c}__sources'] = src
            canaries.append({'id': f'canary_c{c}', 'kind': 'canary', 'client': c, 'shard': owner[c], 'group': 'canary',
                             'forget_rows': int(r['canary_count']), 'source_hash': ids_hash(train['row_id'][src])})
        manifest['alphas'][tag] = {
            'alpha': alpha, 'partition_attempts': attempts, 'client_hashes': [ids_hash(train['row_id'][p]) for p in parts],
            'clients': [{'client': c, 'shard': owner[c], 'rows': len(p), 'labels': np.bincount(train['y'][p], minlength=2).tolist()}
                        for c, p in enumerate(parts)],
            'shards': [{'shard': s, 'clients': m.tolist(), 'rows': int(sum(len(parts[c]) for c in m))} for s, m in enumerate(shards)],
            'pool': {str(c): v for c, v in scorer.pool.items()}, 'f1_val_original': scorer.f1_original,
            'candidates': candidates, 'invalid': invalid, 'requests': selected + canaries,
            'canary_ineligible': [c for c in range(len(parts)) if c not in eligible[:r['max_canary_clients']]]}
        print(f'[{tag}] {len(candidates)} candidates, {len(invalid)} invalid; selected '
              f'{[x["id"] + ":" + x["group"] for x in selected]}; canaries {[x["id"] for x in canaries]}', flush=True)
    np.savez_compressed(out / 'requests.npz', **arrays)
    manifest['seconds'] = time.perf_counter() - started
    write_json(out / 'manifest.json', manifest)
    finalize(out)


# ---------------------------------------------------------------- per-request state
class Request:
    """One deletion: forget rows F at `client`, the original ensemble it is removed from, retained workers."""

    def __init__(self, spec, tag, train, parts, shards, workers, arrays, trigger, cfg, partition_seed):
        self.spec, self.id, self.kind, self.client = spec, spec['id'], spec['kind'], spec['client']
        self.shard = affected_shards(shards, [self.client])[0]
        if self.shard != spec['shard']:
            raise ValueError('Shard assignment differs from the request manifest')
        c = self.client
        if self.kind == 'natural':
            f = arrays[f'{tag}__{self.id}__forget']
            if ids_hash(train['row_id'][f]) != spec['forget_hash']:
                raise ValueError('Forget set differs from the request manifest')
            self.retain = np.setdiff1d(parts[c], f)
            self.forget = (train['X'][f], train['y'][f], train['row_id'][f])
            self.retained = dict(workers)
            self.retained[c] = ClientWorker(c, train['X'][self.retain], train['y'][self.retain], train['row_id'][self.retain])
            self.fraction = spec['fraction']
        else:
            src = arrays[f'{tag}__{self.id}__sources']
            cx, cy, cids, again = canary_rows(train, parts, c, trigger, cfg, partition_seed)
            if not np.array_equal(src, again) or ids_hash(train['row_id'][src]) != spec['source_hash']:
                raise ValueError('Canary rows differ from the request manifest')
            self.retain, self.forget, self.retained, self.fraction = None, (cx, cy, cids), dict(workers), None
        forgotten = set(self.forget[2].tolist())
        self.no_forget_in_retained = not any(forgotten & set(w.row_ids.tolist()) for w in self.retained.values())
        if not self.no_forget_in_retained:
            raise AssertionError(f'{self.id}: forgotten rows reach a retained worker')


def cost(logs, wall, local_epochs):
    return {'rounds': len(logs), 'client_rounds': sum(len(r['clients']) for r in logs),
            'samples_processed': local_epochs * sum(sum(r['counts']) for r in logs),
            'compute_seconds_sum': sum(c['compute_seconds'] for r in logs for c in r['clients']),
            'bytes_up': sum(r['payload_bytes_up'] for r in logs), 'bytes_down': sum(r['payload_bytes_down'] for r in logs),
            'wall_seconds': wall}


def load_model(init_fn, shard, st):
    m = init_fn(shard)
    m.load_state_dict(st)
    return m


def trajectory_signals(states, init_fn, shard, signal):
    pairs = [signal(load_model(init_fn, shard, st)) for st in states]
    return [p[0] for p in pairs], [p[1] for p in pairs]


# ---------------------------------------------------------------- stage: tune
def stage_tune(args):
    cfg, device = setup(args)
    out, req_dir = args.output, args.requests
    if out.exists():
        raise FileExistsError(f'Refusing overwrite: {out}')
    verify_hashes(req_dir)
    req_manifest = json.loads((req_dir / 'manifest.json').read_text())
    if req_manifest['partition_seed'] != cfg['tuning']['pilot_partition_seed'] and not args.smoke:
        raise ValueError('Tuning must run on the pilot partition, not on the reported one')
    data, train, names = load_inputs(args.data, req_manifest['smoke_only'])
    val, P, seed0, fl = data['validation'], req_manifest['partition_seed'], cfg['seeds'][0], cfg['fl']
    dim, hidden, rounds = train['X'].shape[1], cfg['hidden'], fl['rounds']
    init_fn = lambda s: init_shard(dim, hidden, seed0, s)
    base = controller_cfg(cfg, {'rho': 0., 'mu': 0., 'tau_ctrl': 0., 'tau_drop': 0.})
    out.mkdir(parents=True)
    records = []
    for alpha in cfg['alphas']:
        tag = alpha_tag(alpha)
        _, record, parts, shards, arrays = stored_requests(req_dir, cfg, train, alpha)
        workers = client_workers(train, parts)
        original, _ = train_sharded(dim, hidden, shards, workers, fl, rounds, seed0)
        scorer = scorer_for(cfg, original, init_fn, seed0, train['y'], val, workers, shards)
        for spec in [s for s in record['requests'] if s['kind'] == 'natural']:
            req = Request(spec, tag, train, parts, shards, workers, arrays, None, cfg, P)
            members = shards[req.shard]
            ev = eval_indices(parts, members, req.client, req.retain, P, req.shard, cfg['controller']['eval_rows'])
            signal = make_signal_fn(original, req.shard, (train['X'][ev], train['y'][ev]), (val['X'], val['y']))
            states, _ = retrain_trajectory(init_fn(req.shard), shard_workers(members, req.retained), fl, rounds, seed0)
            losses, f1s = trajectory_signals(states, init_fn, req.shard, signal)
            fw = ClientWorker(req.client, *req.forget)
            score = scorer.score(scorer.raw(req.shard, fw, request_seed(seed0, req.client, req.fraction)))
            records.append({'alpha': alpha, 'request': req.id, 'group': spec['group'], 'losses': losses, 'f1s': f1s,
                            'eta0_dynamicfu': initial_eta(score['I'], base), 'f1_original': scorer.f1_original, **score})
            print(f'[tune {tag}/{req.id}] I={score["I"]:.3f} f1 0->{rounds}: {f1s[0]:.4f}->{f1s[-1]:.4f}', flush=True)
    write_json(out / 'signals.json', records)
    t = cfg['tuning']
    rows = []
    for rho, mu, tau_ctrl, tau_drop in itertools.product(*(t['grid'][k] for k in ['rho', 'mu', 'tau_ctrl', 'tau_drop'])):
        params = {'rho': rho, 'mu': mu, 'tau_ctrl': tau_ctrl, 'tau_drop': tau_drop}
        ccfg = controller_cfg(cfg, params)
        row = dict(params)
        for arm in ['dynamicfu', 'dynamicfu_ffree']:
            stops, f1_losses = [], []
            for rec in records:
                eta0 = rec['eta0_dynamicfu'] if arm == 'dynamicfu' else cfg['controller']['ffree_eta0']
                ref = rec['f1_original'] if arm == 'dynamicfu' else None
                stop, _ = replay(rec['losses'], rec['f1s'], ccfg, eta0, ref)
                stops.append(stop)
                f1_losses.append(rec['f1s'][rounds] - rec['f1s'][stop])
            row |= {f'{arm}_mean_rounds': float(np.mean(stops)), f'{arm}_mean_val_f1_loss': float(np.mean(f1_losses)),
                    f'{arm}_max_val_f1_loss': float(np.max(f1_losses))}
        row['feasible'] = all(row[f'{a}_mean_val_f1_loss'] <= t['max_mean_val_f1_loss'] and
                              row[f'{a}_max_val_f1_loss'] <= t['max_val_f1_loss'] for a in ['dynamicfu', 'dynamicfu_ffree'])
        row['objective'] = (row['dynamicfu_mean_rounds'] + row['dynamicfu_ffree_mean_rounds']) / 2
        rows.append(row)
    table = pd.DataFrame(rows)
    table.to_csv(out / 'tuning.csv', index=False)
    feasible = table[table.feasible].sort_values(['objective', 'rho', 'mu', 'tau_ctrl', 'tau_drop'])
    selected = None if feasible.empty else {k: float(feasible.iloc[0][k]) for k in ['rho', 'mu', 'tau_ctrl', 'tau_drop']}
    write_json(out / 'selected.json', {
        'selected': selected, 'feasible_count': int(len(feasible)), 'grid_size': int(len(table)),
        'rule': 'feasible = for both arms, mean over pilot requests of F1_val(full budget) - F1_val(stop) <= '
                f'{t["max_mean_val_f1_loss"]} and max <= {t["max_val_f1_loss"]}; choose min mean rounds of the two '
                'arms, ties by (rho, mu, tau_ctrl, tau_drop) ascending; validation split only',
        'requests_manifest_sha256': digest(req_dir / 'manifest.json'),
        'selected_row': None if feasible.empty else json.loads(feasible.iloc[0].to_json())})
    write_json(out / 'manifest.json', base_manifest(cfg, args.data, args.smoke, device) | {
        'stage': 'tune', 'requests': str(req_dir), 'requests_manifest_sha256': digest(req_dir / 'manifest.json')})
    finalize(out)
    print(json.dumps({'selected': selected, 'feasible': int(len(feasible))}), flush=True)
    if selected is None:
        raise SystemExit('NO_FEASIBLE_CONTROLLER: no grid point satisfies the predeclared rule')


# ---------------------------------------------------------------- stage: run
def stage_run(args):
    cfg, device = setup(args)
    out, req_dir, seed = args.output, args.requests, args.seed
    if out.exists():
        raise FileExistsError(f'Refusing overwrite: {out}')
    verify_hashes(req_dir)
    verify_hashes(args.controller.parent)
    req_manifest = json.loads((req_dir / 'manifest.json').read_text())
    tuned = json.loads(args.controller.read_text())
    if tuned['selected'] is None:
        raise ValueError('Tuning found no feasible controller')
    if not args.smoke and req_manifest['partition_seed'] == cfg['tuning']['pilot_partition_seed']:
        raise ValueError('The run stage reports on the main partition, not the pilot')
    ccfg = controller_cfg(cfg, tuned['selected'])
    data, train, names = load_inputs(args.data, req_manifest['smoke_only'])
    val, test, cal = data['validation'], data['test'], data['calibration']
    P, fl, rounds, rapid = req_manifest['partition_seed'], cfg['fl'], cfg['fl']['rounds'], cfg['rapid_rounds']
    dim, hidden = train['X'].shape[1], cfg['hidden']
    init_fn = lambda s: init_shard(dim, hidden, seed, s)
    trigger = req_manifest['trigger']
    if trigger != trigger_for(data, train, names, cfg):
        raise ValueError('Combination trigger differs from the request manifest')
    backdoor_x = stamp_combination(test['X'][test['y'] == 1], trigger)
    out.mkdir(parents=True)
    (out / 'checkpoints').mkdir()
    (out / 'traces').mkdir()
    write_json(out / 'manifest.json', base_manifest(cfg, args.data, args.smoke, device) | {
        'stage': 'run', 'seed': seed, 'requests': str(req_dir), 'requests_manifest_sha256': digest(req_dir / 'manifest.json'),
        'controller_file': str(args.controller), 'controller_sha256': digest(args.controller),
        'controller': ccfg.__dict__, 'methods': METHODS,
        'timing': 'time.perf_counter; logical workers and shards run sequentially; fixed budgets use the summed '
                  'FedAvg round wall times of the trajectory'})
    rows, curves, contributions, checks, hashes, coverage, timing = [], [], [], {}, {}, [], {}
    started = time.perf_counter()

    def save(name, st):
        torch.save({'state_dict': st, 'input_dim': dim, 'hidden': hidden, 'seed': seed}, out / 'checkpoints' / f'{name}.pt')

    try:
        for alpha in cfg['alphas']:
            tag = alpha_tag(alpha)
            _, record, parts, shards, arrays = stored_requests(req_dir, cfg, train, alpha)
            workers = client_workers(train, parts)
            t = time.perf_counter()
            natural, _ = train_sharded(dim, hidden, shards, workers, fl, rounds, seed)
            timing[f'{tag}_original_seconds'] = time.perf_counter() - t
            t = time.perf_counter()
            global_original, _ = fedavg(model(dim, hidden, seed), [workers[c] for c in sorted(workers)], fl, rounds, seed)
            timing[f'{tag}_global_original_seconds'] = time.perf_counter() - t
            for s, m in enumerate(natural):
                save(f'{tag}_original_shard{s}', state(m))
            save(f'{tag}_global_original', state(global_original))
            t = time.perf_counter()
            natural_scorer = scorer_for(cfg, natural, init_fn, seed, train['y'], val, workers, shards)
            timing[f'{tag}_pool_seconds'] = time.perf_counter() - t
            cal_p = expit(ensemble_margin([margins(m, cal['X']) for m in natural]))
            thresholds = calibrate_mia(cal['y'], cal_p)
            for spec in record['requests']:
                key = f'{tag}/{spec["id"]}'
                try:
                    req = Request(spec, tag, train, parts, shards, workers, arrays, trigger, cfg, P)
                    s_u, members = req.shard, shards[req.shard]
                    if req.kind == 'canary':
                        cx, cy, cids = req.forget
                        c = req.client
                        aug = dict(workers)
                        aug[c] = ClientWorker(c, np.r_[train['X'][parts[c]], cx], np.r_[train['y'][parts[c]], cy],
                                              np.r_[train['row_id'][parts[c]], cids])
                        t = time.perf_counter()
                        shard_model, _ = train_shard(dim, hidden, shard_workers(members, aug), fl, rounds, seed, s_u)
                        original = list(natural)
                        original[s_u] = shard_model
                        scorer = scorer_for(cfg, original, init_fn, seed, np.r_[train['y'], cy], val, aug, shards)
                        timing[f'{key}_canary_original_seconds'] = time.perf_counter() - t
                        save(f'{tag}_{req.id}_original_shard{s_u}', state(shard_model))
                    else:
                        original, scorer = natural, natural_scorer
                    fx, fy, _ = req.forget
                    evaluate_x = {'test': test['X'], 'val': val['X'], 'forget': fx, 'backdoor': backdoor_x}
                    cache = {k: [margins(m, x) for m in original] for k, x in evaluate_x.items()}

                    def ensemble(shard_model, split):
                        zs = list(cache[split])
                        if shard_model is not None:
                            zs[s_u] = margins(shard_model, evaluate_x[split])
                        return expit(ensemble_margin(zs))

                    t = time.perf_counter()
                    full, full_logs = train_sharded(dim, hidden, shards, req.retained, fl, rounds, seed)
                    full_cost = cost([r for l in full_logs for r in l], time.perf_counter() - t, fl['local_epochs'])
                    ev = eval_indices(parts, members, req.client, req.retain, P, s_u, cfg['controller']['eval_rows'])
                    signal = make_signal_fn(original, s_u, (train['X'][ev], train['y'][ev]), (val['X'], val['y']))
                    states, traj_logs = retrain_trajectory(init_fn(s_u), shard_workers(members, req.retained), fl, rounds, seed)
                    losses, f1s = trajectory_signals(states, init_fn, s_u, signal)
                    t = time.perf_counter()
                    raw = scorer.raw(s_u, ClientWorker(req.client, *req.forget), request_seed(seed, req.client, req.fraction))
                    score = scorer.score(raw)
                    eta0 = initial_eta(score['I'], ccfg)
                    contribution_seconds = time.perf_counter() - t
                    arms = {}
                    for arm, arm_eta0, ref in [('dynamicfu', eta0, scorer.f1_original),
                                               ('dynamicfu_ffree', cfg['controller']['ffree_eta0'], None)]:
                        t = time.perf_counter()
                        m, logs, trace = controlled_retrain(init_fn(s_u), shard_workers(members, req.retained), fl, seed,
                                                            ccfg, arm_eta0, ref, signal)
                        wall = time.perf_counter() - t + (contribution_seconds if arm == 'dynamicfu' else 0.)
                        stop, _ = replay(losses, f1s, ccfg, arm_eta0, ref)
                        arms[arm] = {'state': state(m), 'rounds': len(logs), 'trace': trace, 'eta0': arm_eta0,
                                     'cost': cost(logs, wall, fl['local_epochs']), 'replay_stop': stop}
                        write_json(out / 'traces' / f'{tag}_{req.id}_{arm}.json', trace)
                    full_states = [state(m) for m in full]
                    orig_states = [state(m) for m in original]
                    check = {
                        'no_forget_in_retained': req.no_forget_in_retained,
                        'full_equals_affected_trajectory': states_equal(states[rounds], full_states[s_u]),
                        'unaffected_shards_reproduced': all(states_equal(full_states[s], orig_states[s])
                                                            for s in range(len(shards)) if s != s_u),
                        **{f'{a}_is_truncated_trajectory': states_equal(v['state'], states[v['rounds']]) for a, v in arms.items()},
                        **{f'{a}_signals_match_trajectory': [x['loss_ret'] for x in v['trace']] == losses[:v['rounds'] + 1]
                           and [x['f1_val'] for x in v['trace']] == f1s[:v['rounds'] + 1] for a, v in arms.items()},
                        **{f'{a}_replay_matches': v['replay_stop'] == v['rounds'] for a, v in arms.items()}}
                    if req.kind == 'canary':
                        check['full_retrain_equals_natural_original'] = all(
                            states_equal(a, state(b)) for a, b in zip(full_states, natural))
                    checks[key] = check
                    hashes[key] = {'original': [state_hash(x) for x in orig_states], 'full_retrain': [state_hash(x) for x in full_states],
                                   'trajectory': [state_hash(x) for x in states],
                                   **{a: state_hash(v['state']) for a, v in arms.items()}}
                    save(f'{tag}_{req.id}_dynamicfu_shard{s_u}', arms['dynamicfu']['state'])
                    save(f'{tag}_{req.id}_dynamicfu_ffree_shard{s_u}', arms['dynamicfu_ffree']['state'])
                    save(f'{tag}_{req.id}_affected_shard_retrain_shard{s_u}', states[rounds])

                    full_p = {k: expit(ensemble_margin([margins(m, x) for m in full])) for k, x in evaluate_x.items()}
                    per_round = []
                    for r in range(rounds + 1):
                        sm = load_model(init_fn, s_u, states[r])
                        p = {k: ensemble(sm, k) for k in evaluate_x}
                        if r == rounds and any(not np.array_equal(p[k], full_p[k]) for k in p):
                            check['full_predictions_equal_trajectory'] = False
                        per_round.append(p)
                        u = utility(test['y'], p['test'])
                        prefix = traj_logs[:r]
                        curves.append({'alpha': alpha, 'seed': seed, 'request': req.id, 'kind': req.kind, 'group': spec['group'],
                                       'round': r, 'macro_f1': u['macro_f1'], 'fpr': u['fpr'],
                                       'val_macro_f1': utility(val['y'], p['val'])['macro_f1'], 'f1_val_signal': f1s[r],
                                       'loss_ret_signal': losses[r],
                                       'backdoor_p_normal': float(np.mean(1 - p['backdoor'])),
                                       'mae_F_to_full': similarity(p['forget'], full_p['forget'])['prediction_mae'],
                                       'mae_test_to_full': similarity(p['test'], full_p['test'])['prediction_mae'],
                                       **{f'cum_{k}': v for k, v in cost(prefix, sum(x['wall_seconds'] for x in prefix),
                                                                         fl['local_epochs']).items()}})
                    check.setdefault('full_predictions_equal_trajectory', True)
                    probs = {'original': {k: ensemble(None, k) for k in evaluate_x},
                             'full_retrain': full_p,
                             'affected_shard_retrain': per_round[rounds], 'rapid_retrain': per_round[rapid],
                             'dynamicfu': per_round[arms['dynamicfu']['rounds']],
                             'dynamicfu_ffree': per_round[arms['dynamicfu_ffree']['rounds']],
                             'global_original': {k: expit(margins(global_original, x)) for k, x in evaluate_x.items()}}
                    prefix_cost = lambda r: cost(traj_logs[:r], sum(x['wall_seconds'] for x in traj_logs[:r]), fl['local_epochs'])
                    costs = {'full_retrain': full_cost, 'affected_shard_retrain': prefix_cost(rounds),
                             'rapid_retrain': prefix_cost(rapid), 'dynamicfu': arms['dynamicfu']['cost'],
                             'dynamicfu_ffree': arms['dynamicfu_ffree']['cost']}
                    np.savez_compressed(out / f'{tag}_{req.id}_forget_predictions.npz',
                                        **{name: probs[name]['forget'] for name in METHODS})
                    if req.kind == 'natural':
                        mi, ni = matched_indices(fy, test['y'], P + 3)
                    for name in METHODS:
                        p = probs[name]
                        u = utility(test['y'], p['test'])
                        sim_f = similarity(p['forget'], full_p['forget'])
                        row = {'alpha': alpha, 'seed': seed, 'request': req.id, 'kind': req.kind, 'group': spec['group'],
                               'client': req.client, 'fraction': req.fraction, 'forget_rows': len(fy), 'shard': s_u,
                               'method': name, 'macro_f1': u['macro_f1'], 'fpr': u['fpr'], 'pr_auc_ap': u['pr_auc_ap'],
                               'roc_auc': u['roc_auc'], 'val_macro_f1': utility(val['y'], p['val'])['macro_f1'],
                               'backdoor_p_normal': float(np.mean(1 - p['backdoor'])),
                               'mae_F_to_full': sim_f['prediction_mae'], 'disagreement_F_to_full': sim_f['disagreement'],
                               'mae_test_to_full': similarity(p['test'], full_p['test'])['prediction_mae'],
                               'mia_auc_F': mia(fy[mi], p['forget'][mi], test['y'][ni], p['test'][ni], thresholds)['mia_auc']
                               if req.kind == 'natural' else None}
                        if name in arms:
                            row |= {'stop_reason': arms[name]['trace'][-1]['stop_reason'], 'eta0': arms[name]['eta0']}
                        if name == 'dynamicfu':
                            row |= {'I_F': score['I'], 'contribution_seconds': contribution_seconds}
                        if name in costs:
                            c = costs[name]
                            row |= c | {'speedup_samples': full_cost['samples_processed'] / c['samples_processed'],
                                        'speedup_wall': full_cost['wall_seconds'] / c['wall_seconds']}
                        rows.append(row)
                    contributions.append({'alpha': alpha, 'seed': seed, 'request': req.id, 'kind': req.kind,
                                          'group': spec['group'], **{k: v for k, v in raw.items()}, **score, 'eta0': eta0,
                                          'f1_val_original': scorer.f1_original})
                    ok = all(check.values())
                    coverage.append({'alpha': alpha, 'request': req.id, 'status': 'completed' if ok else 'check_failed'})
                    print(f'[{key}] shard {s_u} I={score["I"]:.3f} R_dyn={arms["dynamicfu"]["rounds"]} '
                          f'({arms["dynamicfu"]["trace"][-1]["stop_reason"]}) R_ffree={arms["dynamicfu_ffree"]["rounds"]} '
                          f'checks={"ok" if ok else check}', flush=True)
                except Exception as error:
                    coverage.append({'alpha': alpha, 'request': spec['id'], 'status': 'failed', 'reason': repr(error)})
                    (out / 'error.log').open('a', encoding='utf-8').write(f'{key}\n{traceback.format_exc()}\n')
                    print(f'[{key}] FAILED: {error!r}', flush=True)
                pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
    except Exception:
        (out / 'error.log').open('a', encoding='utf-8').write(traceback.format_exc())
        coverage.append({'alpha': None, 'request': 'pipeline', 'status': 'failed'})
        raise
    finally:
        timing['wall_time_total_seconds'] = time.perf_counter() - started
        pd.DataFrame(rows).to_csv(out / 'metrics.csv', index=False)
        pd.DataFrame(curves).to_csv(out / 'curves.csv', index=False)
        pd.DataFrame(contributions).to_csv(out / 'contributions.csv', index=False)
        pd.DataFrame(coverage).to_csv(out / 'coverage.csv', index=False)
        write_json(out / 'checks.json', checks)
        write_json(out / 'state_hashes.json', hashes)
        write_json(out / 'timings.json', timing)
        all_ok = bool(checks) and all(all(c.values()) for c in checks.values()) and \
            all(c['status'] == 'completed' for c in coverage)
        write_json(out / 'status.json', {'requests': len(checks), 'all_checks_passed': all_ok,
                                         'failed': sum(c['status'] != 'completed' for c in coverage),
                                         'smoke_only': bool(req_manifest['smoke_only'])})
        finalize(out)
    print(json.dumps({'output': str(out), 'seconds': timing['wall_time_total_seconds'], 'all_checks_passed': all_ok}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('stage', choices=['requests', 'tune', 'run'])
    parser.add_argument('--data', type=Path, default=Path('data/processed/connected_v3_seed42'))
    parser.add_argument('--config', type=Path, default=Path('configs/dynamicfu_sample.json'))
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--partition-seed', type=int, default=42, help='requests stage')
    parser.add_argument('--requests', type=Path, help='tune/run: output of the requests stage')
    parser.add_argument('--controller', type=Path, help='run: selected.json of the tune stage')
    parser.add_argument('--seed', type=int, help='run: training seed')
    parser.add_argument('--smoke', action='store_true')
    args = parser.parse_args()
    {'requests': stage_requests, 'tune': stage_tune, 'run': stage_run}[args.stage](args)
