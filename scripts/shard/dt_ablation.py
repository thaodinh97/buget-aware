"""With-DT versus without-DT admission of deletion requests under a time budget (descriptive pilot: no gates, test unused).

  python scripts/shard/dt_ablation.py runs/shard_dt/ablation --seeds 50042 50043 50044 --sleep-check

Workload: the 10-request v2 stream on the S = 4 ensemble ('standard' config), one time budget per request, no byte budget.
Per request every arm chooses between a 20-round retrain (shard_full), a 10-round retrain (shard_fixed) and rejecting.

Arms
  no twin        always_full, always_fixed; reactive = always start, and after the 10th round continue to 20 only if twice
                 the elapsed time fits the budget (a stopwatch, no forecast)
  profiler       static = scripts.shard.shard_dt.plan on the cost table calibrated at fit; static_m1.5 and static_m2 add a
                 fixed safety margin (forecast x margin <= budget)
  twin, history  adaptive = the current DT (per-client EWMA 0.3 after every executed request); last_value = EWMA 1.0;
                 adaptive_inflight = admitted when the adaptive forecast of 10 rounds fits, then the reactive rule
  twin, state    twin_fresh = the static table times the load every client reports before the request. The report is the
                 injected factor itself, so this is idealised telemetry, an upper bound of what state sync can give;
                 twin_stale = the report is 2 requests old; twin_fresh_inflight = twin_fresh admission, then the reactive rule
  reference      oracle = knows the actual seconds of both policies at its own state

Scenarios (load factor of client c during request k, from --scenario-seeds, independent of the training seed)
  stable 1; step = a random half of the clients x3 from request 4 on; burst = the same, requests 4-6 only;
  drift = every client from x1 (request 1) to x3 (request 10), linearly; noisy = LogNormal(0, 0.35) per client and request.

Budgets: T_ref = the static forecast of a 20-round retrain of the largest shard of the initial ensemble; tight / moderate /
loose = 0.25 / 0.5 / 1.0 x T_ref. The multipliers were chosen after looking at the stable per-request costs of seed 50042
and before any arm was run.

Measured and simulated
- Measured: every ensemble state comes from real training through ShardedUnlearner (fork + unlearn), and the compute
  seconds of every client and round come from a real 20-round retrain at that state, run once and shared by all arms.
- Simulated: load. A scenario multiplies each client's measured compute seconds by its factor; the seconds of a policy are
  the sum over its rounds plus the measured server share. Round r of shard_full is round r of shard_fixed (same trainer,
  per-round seeds), so the first 10 rounds of the measured retrain are the trace of shard_fixed.
- --sleep-check replays the always_full trajectory of the first seed under `step` with real sleeps in the workers and
  writes measured against simulated seconds (sleep_check.csv); the weights must equal those of the run without sleeps.

Outcome of a decision, against the oracle at the same state: `violation` (executed, actual seconds > budget), `conservative`
(did less than what fits: 10 rounds instead of 20, or rejected), `ok`. A rejected request stays unserved, there is no retry.

Writes to OUT (must not exist): decisions.csv (one row per decision), trajectories.csv (one row per arm run, with the
validation metrics of its final ensemble), summary.csv, analysis.txt, run.json.
"""
import argparse
import json
import os
import platform
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
import numpy as np
import pandas as pd
import torch

from scripts.shard.replay_v2_stream import request_stream
from scripts.shard.shard_dt import plan, rates
from scripts.shard.shard_service import DATA_DEFAULT
from scripts.shard.shard_v2 import data_content_sha256
from scripts.shard.train_config import CONFIGS, text_sha256
from src.data.federated import partition
from src.eval.utility_breakdown import detector_metrics
from src.fl.fedavg_core import ClientWorker
from src.fu.shard_exact import ForgetRequest, ShardConfig, ShardedUnlearner, assign_shards

SCENARIOS = ('stable', 'step', 'burst', 'drift', 'noisy')
SEEDED = ('step', 'burst', 'noisy')          # scenarios that depend on the scenario seed
BUDGETS = {'tight': .25, 'moderate': .5, 'loose': 1.}
ARMS = {
    'always_full':         {'group': 'no_twin', 'rule': 'full'},
    'always_fixed':        {'group': 'no_twin', 'rule': 'fixed'},
    'reactive':            {'group': 'no_twin', 'rule': 'inflight'},
    'static':              {'group': 'profiler', 'source': 'static'},
    'static_m1.5':         {'group': 'profiler', 'source': 'static', 'margin': 1.5},
    'static_m2':           {'group': 'profiler', 'source': 'static', 'margin': 2.},
    'adaptive':            {'group': 'twin_history', 'source': 'ewma', 'alpha': .3},
    'last_value':          {'group': 'twin_history', 'source': 'ewma', 'alpha': 1.},
    'adaptive_inflight':   {'group': 'twin_history', 'source': 'ewma', 'alpha': .3, 'rule': 'inflight'},
    'twin_fresh':          {'group': 'twin_state', 'source': 'telemetry', 'lag': 0},
    'twin_stale':          {'group': 'twin_state', 'source': 'telemetry', 'lag': 2},
    'twin_fresh_inflight': {'group': 'twin_state', 'source': 'telemetry', 'lag': 0, 'rule': 'inflight'},
    'oracle':              {'group': 'reference', 'rule': 'oracle'},
}
NO_BYTE_LIMIT = 10**15


def load_factors(scenario, clients, requests, seed):
    """factor[k, c]: multiplier of client c's compute seconds during request k (0-based)."""
    if scenario not in SCENARIOS:
        raise ValueError(f'unknown scenario {scenario!r}')
    rng = np.random.default_rng(seed)
    factor = np.ones((requests, clients))
    hit = rng.permutation(clients)[:clients // 2]        # the same clients for step and burst
    if scenario == 'step':
        factor[3:, hit] = 3.
    elif scenario == 'burst':
        factor[3:6, hit] = 3.
    elif scenario == 'drift':
        factor *= np.linspace(1., 3., requests)[:, None]
    elif scenario == 'noisy':
        factor = rng.lognormal(0., .35, size=factor.shape)
    return factor


def round_seconds(trace, factor):
    """Seconds of every round of a traced retrain under the per-client load factors of one request."""
    return sum(seconds * factor[c] for c, seconds in trace['comp'].items()) + trace['server']


def observed_rates(trace, factor, rounds):
    """What the DT measures after `rounds` executed rounds, in the format of shard_dt.rates."""
    clients = [{'client_id': c, 'compute_seconds': float(seconds[r] * factor[c]), 'samples': trace['samples'][c]}
               for c, seconds in trace['comp'].items() for r in range(rounds)]
    return rates([{'server_seconds': float(s)} for s in trace['server'][:rounds]], clients)


class Node:
    """Ensemble state after a sequence of outcomes. Its trace and children come from real training, on demand, once."""

    def __init__(self, fed):
        self.fed, self.trace, self.kids, self.metrics = fed, None, {}, None


class Tree:
    """Every state the arms reach on one stream, keyed by the rounds executed per request (0 = rejected)."""

    def __init__(self, base, stream, validation):
        self.root, self.stream, self.validation = Node(base), stream, validation
        self.full, self.fixed = base.r_base, base.config.shard_fixed_rounds
        self.trainings, self.training_seconds = 0, 0.

    def request(self, k):
        return ForgetRequest(self.stream[k]['client'], self.stream[k]['f'])

    def _train(self, node, k, policy):
        fed = node.fed.fork(policy)
        start = time.perf_counter()
        result = fed.unlearn(self.request(k))
        self.trainings += 1
        self.training_seconds += time.perf_counter() - start
        return fed, result

    def trace(self, node, k):
        """Measured compute seconds per client and round of a full retrain for request k at this state."""
        if node.trace is None:
            fed, result = self._train(node, k, 'shard_full')
            if len(result.round_costs) != self.full:
                raise NotImplementedError('request without a retrain (its shard was removed)')
            comp, samples = {}, {}
            for row in result.client_costs:
                comp.setdefault(row['client_id'], np.zeros(self.full))[row['round'] - 1] = row['compute_seconds']
                samples[row['client_id']] = row['samples']
            node.trace = {'comp': comp, 'samples': samples, 'shard': result.retrained_shard, 'event': result.event,
                          'server': np.array([r['server_seconds'] for r in result.round_costs])}
            node.kids[self.full] = Node(fed)
        return node.trace

    def child(self, node, k, rounds):
        self.trace(node, k)                      # also creates the child of the full retrain
        if rounds not in node.kids:
            node.kids[rounds] = Node(node.fed if rounds == 0 else self._train(node, k, 'shard_fixed')[0])
        return node.kids[rounds]

    def utility(self, node):
        if node.metrics is None:
            m = detector_metrics(self.validation['y'], node.fed.predict_proba(self.validation['X']))
            node.metrics = {'validation_macro_f1': m['macro_f1'], 'validation_fpr': m['fpr']}
        return node.metrics


def run_arm(tree, spec, factors, budget, static):
    """One arm on the stream. Returns (one record per decision, final node, seconds spent forecasting)."""
    node, records, overhead = tree.root, [], 0.
    learned = {'client': dict(static['client']), 'server': static['server']}
    rule, source = spec.get('rule'), spec.get('source')
    for k, item in enumerate(tree.stream):
        trace = tree.trace(node, k)
        per_round = round_seconds(trace, factors[k])
        actual = {0: 0., tree.fixed: float(per_round[:tree.fixed].sum()), tree.full: float(per_round.sum())}
        best = tree.full if actual[tree.full] <= budget else tree.fixed if actual[tree.fixed] <= budget else 0
        forecast, feasible = {}, []
        if source:
            start = time.perf_counter()
            if source == 'telemetry':
                seen = factors[k - spec['lag']] if k >= spec['lag'] else np.ones(factors.shape[1])
                profile = {'client': {c: v * seen[int(c)] for c, v in static['client'].items()}, 'server': static['server']}
            else:
                profile = learned if source == 'ewma' else static
            report = plan(node.fed, tree.request(k), profile, budget / spec.get('margin', 1.), NO_BYTE_LIMIT, True)
            overhead += time.perf_counter() - start
            forecast = {c['rounds']: c['seconds'] for c in report['candidates']}
            feasible = [c['rounds'] for c in report['candidates'] if c['feasible']]
        if rule == 'oracle':
            rounds = best
        elif rule in ('full', 'fixed'):
            rounds = tree.full if rule == 'full' else tree.fixed
        elif rule == 'inflight':                 # the forecast, if any, only decides whether to start at all
            fits = actual[tree.fixed] * tree.full / tree.fixed <= budget
            rounds = 0 if source and not feasible else tree.full if fits else tree.fixed
        else:
            rounds = feasible[0] if feasible else 0
        spent = actual[rounds]
        outcome = 'violation' if spent > budget else 'ok' if rounds == best else 'conservative'
        if rounds and source == 'ewma':
            seen, a = observed_rates(trace, factors[k], rounds), spec['alpha']
            for c, value in seen['client'].items():
                learned['client'][c] = (1 - a) * learned['client'][c] + a * value
            learned['server'] = (1 - a) * learned['server'] + a * seen['server']
        records.append({'step': k + 1, 'client': item['client'], 'kind': item['kind'], 'shard': trace['shard'],
                        'rounds': rounds, 'best_rounds': best, 'outcome': outcome, 'spent_seconds': spent,
                        'policy': {tree.full: 'full', tree.fixed: 'fixed', 0: 'reject'}[rounds],
                        'actual_fixed_seconds': actual[tree.fixed], 'actual_full_seconds': actual[tree.full],
                        'forecast_fixed_seconds': forecast.get(tree.fixed), 'forecast_full_seconds': forecast.get(tree.full)})
        node = tree.child(node, k, rounds)
    return records, node, overhead


class SlowWorker:
    """A client whose local training takes `factor` (>= 1) times longer: it sleeps for the extra time. Same weights."""

    def __init__(self, inner, factor):
        self.inner, self.factor, self.client_id = inner, max(1., float(factor)), inner.client_id

    def train(self, model, config, seed):
        update, count, log = self.inner.train(model, config, seed)
        time.sleep((self.factor - 1) * log['compute_seconds'])
        return update, count, {**log, 'compute_seconds': self.factor * log['compute_seconds']}


def slow_factory(fed, factor):
    tr = fed.train
    return lambda c, rows: SlowWorker(ClientWorker(c, tr['X'][rows], tr['y'][rows], tr['row_id'][rows], fed.config.device),
                                      factor[c])


def sleep_check(tree, factors):
    """The always_full trajectory with real sleeps in the workers: measured against simulated seconds per request."""
    node, records = tree.root, []
    for k in range(len(tree.stream)):
        simulated = float(round_seconds(tree.trace(node, k), factors[k]).sum())
        fed = node.fed.fork('shard_full')
        fed.worker_factory = slow_factory(fed, factors[k])
        result = fed.unlearn(tree.request(k))
        plain, j = node.kids[tree.full].fed, result.retrained_shard
        same = all(torch.equal(a, b) for a, b in zip(fed.shards[j].model.state_dict().values(),
                                                     plain.shards[j].model.state_dict().values()))
        measured = float(sum(r['wall_seconds'] for r in result.round_costs))
        records.append({'step': k + 1, 'simulated_seconds': simulated, 'measured_seconds': measured,
                        'relative_error': measured / simulated - 1, 'weights_equal': same})
        node = node.kids[tree.full]
    return records


def summarise(decisions, by):
    """Shares of the outcomes and of the executed policies, the mean overshoot of the violations and the forecast error."""
    out = []
    for key, g in decisions.groupby(by, sort=False):
        key = key if isinstance(key, tuple) else (key,)
        late, known = g[g.outcome == 'violation'], g[g.forecast_full_seconds.notna()]
        out.append({**dict(zip(by, key)), 'decisions': len(g),
                    'ok': (g.outcome == 'ok').mean(), 'violation': (g.outcome == 'violation').mean(),
                    'conservative': (g.outcome == 'conservative').mean(),
                    'full': (g.policy == 'full').mean(), 'fixed': (g.policy == 'fixed').mean(),
                    'rejected': (g.policy == 'reject').mean(),
                    'overshoot_mean': (late.spent_seconds / late.budget_seconds - 1).mean() if len(late) else np.nan,
                    'forecast_mape': ((known.forecast_full_seconds - known.actual_full_seconds).abs()
                                      / known.actual_full_seconds).mean() if len(known) else np.nan})
    return pd.DataFrame(out)


def analysis(decisions, trajectories, checks):
    pct = lambda frame, cols: frame.assign(**{c: (100 * frame[c]).round(1) for c in cols})
    shares = ['ok', 'violation', 'conservative', 'full', 'fixed', 'rejected', 'overshoot_mean', 'forecast_mape']
    lines = ['Outcome shares in percent. overshoot_mean: mean excess over the budget of the violations. forecast_mape: mean '
             'absolute error of the forecast of the 20-round retrain. Descriptive; decisions of one run are not independent.', '',
             '== all scenarios and budgets ==',
             pct(summarise(decisions, ['group', 'arm']), shares).to_string(index=False)]
    for scenario in decisions.scenario.unique():
        lines += ['', f'== scenario {scenario} ==',
                  pct(summarise(decisions[decisions.scenario == scenario], ['arm']), shares).to_string(index=False)]
    for budget in decisions.budget.unique():
        lines += ['', f'== budget {budget} ==',
                  pct(summarise(decisions[decisions.budget == budget], ['arm']), shares).to_string(index=False)]
    final = trajectories.groupby('arm', sort=False)[['served', 'validation_macro_f1', 'validation_fpr', 'spent_seconds',
                                                      'forecast_seconds']].mean().round(5)
    lines += ['', '== per run of 10 requests, mean over all runs ==', final.to_string()]
    if checks is not None:
        lines += ['', '== sleep check: always_full under step, measured with real sleeps against simulated ==',
                  checks.round(4).to_string(index=False)]
    return '\n'.join(lines) + '\n'


def experiment(data, seeds, scenario_seeds, shards=4, train_config='standard', clients=20, check=False, log=print):
    """Returns (decisions, trajectories, sleep check or None, per-seed info)."""
    train = data['train']
    parts = partition(train['y'], 42, .5, clients)
    stream = request_stream(train['y'], parts)
    config = ShardConfig.from_train_config(CONFIGS[train_config])
    decisions, trajectories, info, checks = [], [], [], None
    for seed in seeds:
        base = ShardedUnlearner(train, data['validation'], parts, assign_shards(shards, clients), seed, 'shard_full', config,
                                measure_payload=False)
        fit = base.fit()
        static = rates(fit['round_costs'], fit['client_costs'])
        reference = max(base.r_base * (sum(static['client'][str(c)] * len(parts[c]) for c in shard) + static['server'])
                        for shard in base.assignment)
        tree = Tree(base, stream, data['validation'])
        for scenario in SCENARIOS:
            for scenario_seed in (scenario_seeds if scenario in SEEDED else scenario_seeds[:1]):
                factors = load_factors(scenario, clients, len(stream), scenario_seed)
                for budget_name, share in BUDGETS.items():
                    budget = share * reference
                    for arm, spec in ARMS.items():
                        records, node, overhead = run_arm(tree, spec, factors, budget, static)
                        head = {'seed': seed, 'scenario': scenario, 'scenario_seed': scenario_seed, 'budget': budget_name,
                                'budget_seconds': budget, 'group': spec['group'], 'arm': arm}
                        decisions += [{**head, **r} for r in records]
                        trajectories.append({**head, 'served': sum(r['rounds'] > 0 for r in records),
                                             'spent_seconds': sum(r['spent_seconds'] for r in records),
                                             'forecast_seconds': overhead, **tree.utility(node)})
            log(f'seed {seed} {scenario}: {tree.trainings} retrains so far, {tree.training_seconds:.0f} s')
        if check and checks is None:
            checks = pd.DataFrame(sleep_check(tree, load_factors('step', clients, len(stream), scenario_seeds[0])))
        info.append({'seed': seed, 'reference_seconds': reference, 'retrains': tree.trainings,
                     'retrain_seconds': tree.training_seconds, 'static_profile': static,
                     'initial': tree.utility(tree.root)})
    return pd.DataFrame(decisions), pd.DataFrame(trajectories), checks, info


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('output', type=Path)
    parser.add_argument('--data', type=Path, default=ROOT / DATA_DEFAULT)
    parser.add_argument('--seeds', type=int, nargs='+', default=[50042, 50043, 50044])
    parser.add_argument('--scenario-seeds', type=int, nargs='+', default=[301, 302, 303])
    parser.add_argument('--shards', type=int, default=4)
    parser.add_argument('--sleep-check', action='store_true')
    args = parser.parse_args()
    from src.data.loading import load_data
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = load_data(args.data)
    args.output.mkdir(parents=True, exist_ok=False)
    hashes = {f: text_sha256(ROOT / f) for f in ('scripts/shard/dt_ablation.py', 'scripts/shard/shard_dt.py',
                                                 'src/fu/shard_exact.py', 'src/fu/shard_cost.py')}
    started = time.perf_counter()
    decisions, trajectories, checks, info = experiment(data, args.seeds, args.scenario_seeds, args.shards,
                                                       check=args.sleep_check, log=lambda m: print(m, file=sys.stderr, flush=True))
    decisions.to_csv(args.output / 'decisions.csv', index=False)
    trajectories.to_csv(args.output / 'trajectories.csv', index=False)
    summarise(decisions, ['scenario', 'budget', 'group', 'arm']).to_csv(args.output / 'summary.csv', index=False)
    if checks is not None:
        checks.to_csv(args.output / 'sleep_check.csv', index=False)
    text = analysis(decisions, trajectories, checks)
    (args.output / 'analysis.txt').write_text(text, encoding='utf-8')
    run = {'design': {'scenarios': SCENARIOS, 'budgets': BUDGETS, 'arms': ARMS, 'shards': args.shards, 'train_config': 'standard',
                      'seeds': args.seeds, 'scenario_seeds': args.scenario_seeds},
           'data_content_sha256': data_content_sha256(data),
           'sha256_at_start': hashes,
           'machine': {'platform': platform.platform(), 'python': platform.python_version(), 'torch': torch.__version__,
                       'numpy': np.__version__, 'threads': torch.get_num_threads(), 'cpus': os.cpu_count()},
           'seeds': info, 'wall_seconds': time.perf_counter() - started,
           'load': 'simulated: measured client compute seconds x scenario factor', 'test_split': 'unused'}
    (args.output / 'run.json').write_text(json.dumps(run, indent=1), encoding='utf-8')
    print(text)


if __name__ == '__main__':
    main()
