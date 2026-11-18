"""The mid-course adjustment run online: the twin drives the component's shard_checkpoint policy (direction 2 of
docs/dt_extensions.md).

  python scripts/dt/run_midcourse_online.py OUT --seed 50042 --family spread --env slow_clients --budget-level mid

The ensemble is trained once without stress; the 20 requests of the family then run under the chosen stress environment.
Before each request the twin (Adaptive, q90 margin, src/dt/planner.Twin in midcourse mode) forecasts shard_fixed; if
forecast x margin fits the budget it starts the request with policy shard_checkpoint, passing the budget and its margin on the
extrapolation (factor - 1); the component then decides at round 10 from the measured round time. Otherwise the request is
escalated and run with shard_full outside the budget. The twin learns from every operation.

Online, a request stopped at the checkpoint never runs its last 10 rounds, so whether shard_full would have fit stays unknown
here; the paired offline evaluation (scripts/dt/evaluate_stress.py) measures that. This run shows the policy working inside
the loop and checks the end state bit for bit.

Every row of decisions.jsonl also carries the twin's own cost around that request: dt_seconds (state: reading the shard's
client counts from the live unlearner; plan: forecasts, margins, decision; sync: folding the operation's cost records into
the twin) and the telemetry bytes of the operation against its raw model payload (src/dt/overhead.py). final.json sums them.
--margin-mode rescored uses the re-scored margin instead of the plain one.

Writes OUT/run.json, OUT/decisions.jsonl, OUT/final.json. Validation only.
"""
import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import torch

from scripts.dt.machine import prefer_performance
from scripts.dt.run_stress_trajectories import shard_counts, stream_requests
from scripts.shard.shard_v2 import DATA_DEFAULT, DATA_SHA256, data_content_sha256
from scripts.shard.stream_v3 import FAMILIES
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.dt import planner
from src.dt.overhead import Stopwatch, telemetry_bytes
from src.dt.stress import ENVIRONMENTS, STRESS, environment, install
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner, assign_shards, _py


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    parser.add_argument('--seed', type=int, required=True)
    parser.add_argument('--family', choices=FAMILIES, required=True)
    parser.add_argument('--env', choices=ENVIRONMENTS, required=True)
    parser.add_argument('--budget-level', choices=list(planner.BUDGET_LEVELS), required=True)
    parser.add_argument('--margin-mode', choices=('ratio', 'rescored'), default='ratio')
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--data', type=Path, default=ROOT/DATA_DEFAULT)
    parser.add_argument('--threads', type=int, default=2)
    parser.add_argument('--train-config', default='standard', help=argparse.SUPPRESS)   # tests only
    args = parser.parse_args(argv)
    from src.data.loading import load_data
    args.output.mkdir(parents=True, exist_ok=False)
    process = prefer_performance()
    torch.set_num_threads(args.threads); torch.use_deterministic_algorithms(True)
    data = load_data(args.data); validation = data['validation']
    parts = partition(data['train']['y'], 42, .5, 20)
    config = ShardConfig.from_train_config(CONFIGS[args.train_config], min_clients_design=4, merge_below=3)
    base = ShardedUnlearner(data['train'], validation, parts, assign_shards(args.shards, 20), args.seed, 'shard_full', config)
    fit = base.fit()
    profile = planner.rates(fit['round_costs'], fit['client_costs'])
    history = planner.fit_ratios(fit['round_costs'], fit['client_costs'], profile, config.shard_fixed_rounds)
    full, fixed = config.fl['rounds'], config.shard_fixed_rounds
    largest = max(sum(r['wall_seconds'] for r in fit['round_costs'] if r['shard'] == j) for j in range(args.shards))
    budget = planner.BUDGET_LEVELS[args.budget_level]*largest
    cfg = planner.PlannerConfig('adaptive', .9, 0, 'midcourse', margin_mode=args.margin_mode, full_rounds=full,
                                fixed_rounds=fixed)
    twin = planner.Twin(cfg, profile, history)
    requests = stream_requests(base, args.family)
    fed = base.fork('shard_full')
    controller = install(fed, environment(args.env, fed.assignment_initial))
    (args.output/'run.json').write_text(json.dumps(_py({
        'seed': args.seed, 'family': args.family, 'env': args.env, 'stress': STRESS, 'budget_level': args.budget_level,
        'budget_seconds': budget, 'largest_shard_full_seconds': largest, 'planner': cfg.name, 'data_content_sha256':
        data_content_sha256(data), 'data_is_v2': data_content_sha256(data) == DATA_SHA256, 'train_config': args.train_config,
        'process': process}), indent=1), encoding='utf-8')
    totals = {'state': 0., 'plan': 0., 'sync': 0., 'fu_round_seconds': 0.}
    for k, r in enumerate(requests):
        controller.step = r['step']
        watch = Stopwatch()                                  # the twin's own time around this request (src/dt/overhead.py)
        with watch('state'):
            counts = shard_counts(fed, r['shard'], r['client'], len(r['f']))
        with watch('plan'):
            f_full, f_fixed = twin.forecasts(counts, r['shard'])
            m, m2 = twin.factors()
            decision = twin.start(f_full, f_fixed, budget)
        started = time.perf_counter()
        if decision == 'start':
            result = fed.unlearn(ForgetRequest(r['client'], r['f']), 'shard_checkpoint', budget_seconds=budget, margin=m2 - 1)
            choice = 'full' if result.details['continued'] else 'fixed'
        else:
            result = fed.unlearn(ForgetRequest(r['client'], r['f']), 'shard_full')
            choice = 'reject'
        walls = [x['wall_seconds'] for x in result.round_costs]
        spent, predicted = float(sum(walls)), (f_fixed if choice == 'fixed' else f_full)
        extra = planner.extrapolation_ratio(sum(walls[:fixed]), spent, full, fixed) if len(walls) == full else None
        with watch('sync'):                                  # lag 0: the observation is folded in before the next request
            twin.record(k, (result.round_costs, result.client_costs), spent/predicted, extra)
            twin.advance(k + 1)
        for key, value in watch.seconds.items():
            totals[key] += value
        totals['fu_round_seconds'] += spent
        met = detector_metrics(validation['y'], fed.predict_proba(validation['X']))
        row = {'step': r['step'], 'shard': r['shard'], 'decision': decision, 'choice': choice, 'policy': result.policy,
               'rounds': result.cost['rounds'], 'forecast_full': f_full, 'forecast_fixed': f_fixed, 'factor': m,
               'factor_extrapolation': m2, 'seconds': spent, 'seconds_until_checkpoint': float(sum(walls[:fixed])),
               'budget_seconds': budget, 'violation': choice != 'reject' and spent > budget, 'details': result.details,
               'warnings': [w['code'] for w in result.structured_warnings], 'exactness_scope': result.exactness_scope,
               'exact': result.exact, 'structure_ok': result.structure_ok, 'validation_macro_f1': met['macro_f1'],
               'validation_fpr': met['fpr'], 'operation_wall_seconds': time.perf_counter() - started,
               'dt_seconds': dict(watch.seconds), 'dt_overhead_seconds': sum(watch.seconds.values()),
               'telemetry_bytes': telemetry_bytes(result.round_costs, result.client_costs, fixed),
               'payload_bytes_raw': result.cost['train_bytes_raw'], 'payload_pack_seconds': result.cost['payload_pack_seconds']}
        with open(args.output/'decisions.jsonl', 'a', encoding='utf-8') as handle:
            handle.write(json.dumps(_py(row)) + '\n')
        print(json.dumps(_py({k: row[k] for k in ('step', 'shard', 'choice', 'rounds', 'seconds', 'violation')})), flush=True)
    del fed.worker_factory                                   # verify without the injected delays (weights do not depend on them)
    check = fed.verify_bit_exact()
    met = detector_metrics(validation['y'], fed.predict_proba(validation['X']))
    (args.output/'final.json').write_text(json.dumps(_py({'verify_bit_exact': check, 'validation_macro_f1': met['macro_f1'],
        'validation_fpr': met['fpr'], 'shard_rounds': {j: e.rounds for j, e in enumerate(fed.shards) if e},
        'dt_overhead': {**totals, 'dt_seconds': totals['state'] + totals['plan'] + totals['sync'],
                        'share_of_fu_round_seconds': (totals['state'] + totals['plan'] + totals['sync'])/totals['fu_round_seconds']}}),
        indent=1), encoding='utf-8')
    print(json.dumps(_py({'bit_exact': check['bit_exact'], 'structure_ok': check['structure_ok']})))
    if not (check['bit_exact'] and check['structure_ok'] and check['included_equals_active']):
        sys.exit(1)


if __name__ == '__main__':
    main()
