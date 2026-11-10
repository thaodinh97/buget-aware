"""Replay the v2 deletion stream through the component (src/fu/shard_exact.py) and record the cost input of the Digital Twin.

  python scripts/shard/replay_v2_stream.py OUT --shards 4 --seed 50042 --method utility_stop --compare results/shard_v2/stream_20c

Same scenario as scripts/shard/run_shard_stream_v2.py (data, 20-client partition, shard assignment, request stream, 'standard'
train config, >= 4 clients per shard, merge below 3), for one S, one training seed and one policy. Writes to OUT (must not exist):
  steps.csv         per request: event, exactness, calibration metrics (as the v2 runner reports them) and the cost summary
  round_costs.csv   per FedAvg round: clients, samples, wall / compute / server seconds, bytes
  client_costs.csv  per client and round: samples, optimizer steps, loss, compute seconds, bytes down / up (raw and measured)
  fit.json          cost of training the base ensemble, client state, inference benchmark
  compare.json      with --compare: this replay against the stored v2 steps.csv of the same (S, seed, policy)
With --compare the exit status is 1 unless every discrete quantity (events, round counts, rows x rounds, bytes, policy outputs)
is equal and the calibration metrics differ by at most --tolerance (default 0: bit-equal, expected on the machine of the v2 run).
Test split unused.
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
from scipy.special import expit

from scripts.shard.shard_v2 import DATA_DEFAULT, DATA_SHA256, DESIGNS, data_content_sha256
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import POLICIES, ForgetRequest, ShardConfig, ShardedUnlearner, assign_shards, _py

DISCRETE = ['kind', 'event', 'exact', 'structure_ok', 'shards_included', 'rows_x_rounds', 'rounds', 'bytes',
            'phase1_rounds', 'fallback_rounds', 'condition_met', 'contribution', 'R0', 'stop_reason', 'pending']
METRICS = ['cal_macro_f1', 'cal_fpr', 'cal_recall_attack', 'client_f1_min', 'client_f1_mean']
COSTS = ['clients_selected', 'client_rounds', 'bytes_down_raw', 'bytes_up_raw', 'bytes_down_packed', 'bytes_up_packed',
         'compute_seconds_sum', 'server_aggregate_seconds', 'policy_eval_seconds', 'payload_pack_seconds',
         'makespan_seconds_simulated', 'wall_seconds_total', 'rss_bytes_peak_sampled', 'models_stored', 'inference_forward_passes']


def request_stream(y, parts, requests=10, fraction=.3, seed=99):
    """The request stream of the v2 runners, from metadata only: odd steps delete a whole client, even steps a random
    `fraction` of a client's remaining rows; the client is drawn with probability proportional to its remaining rows."""
    rng = np.random.default_rng(seed); alive = np.ones(len(y), bool); present = set(range(len(parts))); stream = []
    for k in range(requests):
        rem = [(c, parts[c][alive[parts[c]]]) for c in sorted(present)]; w = np.array([len(r) for _, r in rem], float)
        c, rows = rem[rng.choice(len(rem), p=w/w.sum())]; kind = 'client' if k % 2 == 0 else 'sample'
        f = rows if kind == 'client' else np.sort(rng.choice(rows, max(1, int(len(rows)*fraction)), replace=False))
        if kind == 'client' or len(f) == len(rows): present.discard(c); kind = 'client'
        alive[f] = False; stream.append({'step': k + 1, 'client': int(c), 'kind': kind, 'rows': int(len(f)), 'f': f})
    return stream


def report(fed, train, calibration):
    """Calibration metrics of the ensemble and macro-F1 on every present client's retained rows, as in the v2 runner."""
    out = {f'cal_{k}': v for k, v in detector_metrics(calibration['y'], expit(fed.margin(calibration['X']))).items()
           if k in ('macro_f1', 'fpr', 'recall_attack')}
    margin = fed.margin(train['X'])
    f1 = [detector_metrics(train['y'][p[fed.alive[p]]], expit(margin[p[fed.alive[p]]]))['macro_f1']
          for c, p in enumerate(fed.parts) if c in fed.present]
    out.update(client_f1_min=float(np.nanmin(f1)), client_f1_mean=float(np.nanmean(f1)))
    return out


def replay(data, shards, seed, policy, config, clients=20, alpha=.5, partition_seed=42):
    """Returns (steps, fit, results, unlearner): one row per request (plus the backlog row of drop_backfill)."""
    train, calibration = data['train'], data['calibration']
    parts = partition(train['y'], partition_seed, alpha, clients)
    stream = request_stream(train['y'], parts)
    fed = ShardedUnlearner(train, data['validation'], parts, assign_shards(shards, clients), seed, policy, config)
    fit = fed.fit()
    fit.update(clients=fed.client_state(), inference=fed.benchmark_inference(calibration['X']), initial=report(fed, train, calibration))
    steps, results = [], []
    calls = [lambda r=r: fed.unlearn(ForgetRequest(r['client'], r['f'])) for r in stream]
    if policy == 'drop_backfill':
        calls.append(fed.finish_backlog)
    for number, call in enumerate(calls, 1):
        result = call(); total = result.cost['cumulative_unlearning']
        steps.append({'S': shards, 'seed': seed, 'policy': policy, 'step': number, 'kind': result.kind, 'event': result.event,
                      'client': result.client, 'request_rows': result.rows, 'exact': result.exact, 'structure_ok': result.structure_ok,
                      'exactness_scope': result.exactness_scope, 'shards_included': result.shards_included,
                      'rows_x_rounds': total['rows_x_rounds'], 'rounds': total['rounds'], 'bytes': total['train_bytes_raw'],
                      **report(fed, train, calibration), **result.details,
                      'step_rounds': result.cost['rounds'], 'step_rows_x_rounds': result.cost['rows_x_rounds'],
                      **{k: result.cost[k] for k in COSTS},
                      'warnings': [w['code'] for w in result.structured_warnings]})
        results.append(result)
    return steps, fit, results, fed


def read_steps(path):
    """A steps.csv with its float columns parsed by Python's float(), which round-trips exactly on every pandas version."""
    exact = lambda text: float(text) if text else float('nan')
    return pd.read_csv(path, converters={key: exact for key in METRICS + ['contribution']})


def _same(a, b):
    """Equality of a replay value and a value read back from a v2 steps.csv (missing = None / NaN on both sides)."""
    if a is None or (isinstance(a, float) and np.isnan(a)): return bool(pd.isna(b))
    if pd.isna(b): return False
    if isinstance(a, (bool, np.bool_)) or isinstance(b, (bool, np.bool_)): return bool(a) == bool(b)
    if isinstance(a, (int, float, np.number)): return float(a) == float(b)
    return str(a) == str(b)


def compare(steps, reference, tolerance=0.):
    """This replay against the rows of a v2 steps.csv with the same S, seed and policy. The v2 backlog row of drop_backfill
    carries no event and no policy outputs, so those are skipped there."""
    per_step, mismatches, worst = [], [], 0.
    for row in steps:
        ref = reference[(reference.S == row['S']) & (reference.seed == row['seed']) & (reference.policy == row['policy'])
                        & (reference.step == row['step'])]
        if len(ref) != 1:
            mismatches.append({'step': row['step'], 'field': 'row', 'replay': 'present', 'v2': f'{len(ref)} rows'})
            continue
        ref = ref.iloc[0]
        skip = ('event', 'pending') if row['kind'] == 'backlog_done' else ()
        for key in DISCRETE:
            if key in row and key not in skip and not _same(row[key], ref.get(key)):
                mismatches.append({'step': row['step'], 'field': key, 'replay': row[key], 'v2': ref.get(key)})
        diffs = {key: abs(float(row[key]) - float(ref[key])) for key in METRICS}
        worst = max(worst, *diffs.values())
        per_step.append({'step': row['step'], **diffs})
    return {'pass': not mismatches and worst <= tolerance and len(per_step) == len(steps), 'discrete_equal': not mismatches,
            'metrics_max_abs_diff': worst, 'metrics_bit_equal': worst == 0., 'tolerance': tolerance,
            'steps_compared': len(per_step), 'mismatches': mismatches, 'per_step': per_step}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    parser.add_argument('--shards', type=int, required=True)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--method', choices=POLICIES, required=True)
    parser.add_argument('--data', type=Path, default=ROOT/DATA_DEFAULT)
    parser.add_argument('--compare', type=Path, help='directory with the steps.csv of run_shard_stream_v2.py')
    parser.add_argument('--tolerance', type=float, default=0.)
    parser.add_argument('--threads', type=int, default=2)
    args = parser.parse_args(argv)
    from src.data.loading import load_data
    args.output.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    data = load_data(args.data); content = data_content_sha256(data)
    if args.compare and content != DATA_SHA256:
        raise ValueError(f'data content {content} != {DATA_SHA256}: not the data of the v2 runs')
    design = DESIGNS['v2']
    config = ShardConfig.from_train_config(CONFIGS[design['train_config']], min_clients_design=design['min_clients_design'],
                                           merge_below=design['merge_below'])
    steps, fit, results, fed = replay(data, args.shards, args.seed, args.method, config)
    pd.DataFrame(steps).to_csv(args.output/'steps.csv', index=False)
    tag = lambda result, rows: [{'step': result.step, 'kind': result.kind, **r} for r in rows]
    pd.DataFrame([r for res in results for r in tag(res, res.round_costs)]).to_csv(args.output/'round_costs.csv', index=False)
    pd.DataFrame([r for res in results for r in tag(res, res.client_costs)]).to_csv(args.output/'client_costs.csv', index=False)
    fit_summary = {k: v for k, v in fit.items() if k not in ('round_costs', 'client_costs')}
    (args.output/'fit.json').write_text(json.dumps(_py({'shards': args.shards, 'seed': args.seed, 'policy': args.method,
        'data_content_sha256': content, 'data_is_v2': content == DATA_SHA256, 'config': config.to_json(),
        'torch': torch.__version__, 'threads': args.threads, 'bit_exact_end': fed.verify_bit_exact(), **fit_summary}), indent=1), encoding='utf-8')
    pd.DataFrame(fit['client_costs']).to_csv(args.output/'fit_client_costs.csv', index=False)
    last = steps[-1]
    print(json.dumps(_py({'output': args.output, 'steps': len(steps), 'final_cal_macro_f1': last['cal_macro_f1'],
                          'final_cal_fpr': last['cal_fpr'], 'rounds': last['rounds'], 'rows_x_rounds': last['rows_x_rounds']})))
    if args.compare:
        outcome = compare(steps, read_steps(args.compare/'steps.csv'), args.tolerance)
        (args.output/'compare.json').write_text(json.dumps(_py(outcome), indent=1), encoding='utf-8')
        print(json.dumps(_py({k: outcome[k] for k in ('pass', 'discrete_equal', 'metrics_max_abs_diff', 'steps_compared')})))
        print(*[json.dumps(_py(m)) for m in outcome['mismatches'][:10]], sep='\n')
        if not outcome['pass']:
            sys.exit(1)


if __name__ == '__main__':
    main()
