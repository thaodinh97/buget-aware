"""Budget-aware admission for fixed-round shard FU; costs cover sequential FedAvg rounds."""
import argparse
import json
import math
import tempfile
from pathlib import Path

import numpy as np
import torch

from scripts.shard.shard_service import DATA_DEFAULT, load_state, read_request, write_costs
from src.data.loading import load_data
from src.eval.utility_breakdown import detector_metrics
from src.fu.shard_exact import structure_step


def rates(rounds, clients):
    # Fit per-client compute seconds per retained sample, plus server seconds per round.
    totals = {}
    for row in clients:
        cid = str(row['client_id'])
        seconds, samples = totals.get(cid, (0., 0))
        totals[cid] = seconds + row['compute_seconds'], samples + row['samples']
    return {'client': {c: s / n for c, (s, n) in totals.items()},
            'server': sum(r['server_seconds'] for r in rounds) / len(rounds)}


def plan(fed, request, profile, seconds, byte_limit, allow_fixed):
    if fed.pending:
        raise ValueError('Finish pending backfill before budgeted fixed-round FU')
    c = request.client
    if c not in fed.present:
        raise ValueError('Client already removed or unknown')
    held = fed.parts[c][fed.alive[fed.parts[c]]]
    forget = held if request.rows is None else request.rows
    if not len(forget) or len(np.unique(forget)) != len(forget) or not np.isin(forget, held).all():
        raise ValueError('Invalid or previously deleted rows')
    kind = 'client' if len(forget) == len(held) else 'sample'
    layout = structure_step(fed.assignment, c, kind, fed.config.merge_below)
    shard = layout['retrain']
    clients = [] if shard is None else layout['assign_after'][shard]
    counts = {i: int(fed.alive[fed.parts[i]].sum()) - (len(forget) if i == c else 0) for i in clients}
    duration = sum(profile['client'][str(i)] * n for i, n in counts.items())
    if clients:
        duration += profile['server']
    candidates = []
    options = [('shard_full', fed.r_base)]
    if allow_fixed:
        options.append(('shard_fixed', fed.config.shard_fixed_rounds))
    for policy, rounds in options:
        rounds = rounds if clients else 0
        sec = rounds * duration
        size = rounds * 2 * len(clients) * fed.model_bytes
        candidates.append(dict(policy=policy, rounds=rounds, seconds=sec, raw_bytes=size,
                               feasible=sec <= seconds and size <= byte_limit))
    return dict(kind=kind, event=layout['event'], clients=clients, candidates=candidates)


def execute(data, original, request, destination, mode, seconds, byte_limit, allow_fixed=False):
    if seconds < 0 or not math.isfinite(seconds) or byte_limit < 0:
        raise ValueError('Budgets must be finite and nonnegative')
    fed, layout = load_state(data, original)
    profile_path = original / 'twin.json'
    if profile_path.exists():
        profiles = json.loads(profile_path.read_text())
    else:
        import pandas as pd
        if json.loads((original / 'result.json').read_text())['operation'] != 'init':
            raise ValueError('Start from an init checkpoint or the output of an accepted DT request')
        base = rates(pd.read_csv(original / 'round_costs.csv').to_dict('records'),
                     pd.read_csv(original / 'client_costs.csv').to_dict('records'))
        profiles = {'static': base, 'adaptive': dict(client=base['client'].copy(), server=base['server'])}
    report = plan(fed, request, profiles[mode], seconds, byte_limit, allow_fixed)
    choice = next((x for x in report['candidates'] if x['feasible']), None)
    if choice:
        other = 'adaptive' if mode == 'static' else 'static'
        alternative = plan(fed, request, profiles[other], seconds, byte_limit, allow_fixed)
        paired = next(x for x in alternative['candidates'] if x['policy'] == choice['policy'])
        report['predicted_seconds_same_policy'] = {mode: choice['seconds'], other: paired['seconds']}
    report.update(mode=mode, budget_seconds=seconds, budget_raw_bytes=byte_limit,
                  status='NO_FEASIBLE_PLAN', selected=choice)
    if destination.exists():
        raise FileExistsError(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=destination.parent) as temp:
        output = Path(temp) / 'state'
        if choice:
            result = fed.unlearn(request, choice['policy'], output)
            summary = result.to_json()
            summary['final_checkpoint'] = destination.as_posix()
            rounds = summary.pop('round_costs')
            clients = summary.pop('client_costs')
            write_costs(output, {'operation': 'unlearn', **summary, 'clients': fed.client_state()}, rounds, clients, layout)
            actual = sum(r['wall_seconds'] for r in rounds)
            size = result.cost['train_bytes_raw']
            report.update(status='COMPLETED', actual_seconds=actual, actual_raw_bytes=size,
                          budget_violated=actual > seconds or size > byte_limit,
                          exact=result.exact, structure_ok=result.structure_ok)
            validation = data['validation']
            metrics = detector_metrics(validation['y'], fed.predict_proba(validation['X']))
            report['validation_macro_f1'] = metrics['macro_f1']
            report['validation_fpr'] = metrics['fpr']
            if rounds:
                observed = rates(rounds, clients)
                adaptive = profiles['adaptive']
                for c, value in observed['client'].items():
                    adaptive['client'][c] = .7 * adaptive['client'][c] + .3 * value
                adaptive['server'] = .7 * adaptive['server'] + .3 * observed['server']
            (output / 'twin.json').write_text(json.dumps(profiles))
        else:
            output.mkdir()
        (output / 'dt_report.json').write_text(json.dumps(report, indent=2))
        output.rename(destination)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('original-run', 'forget-request', 'output'):
        parser.add_argument('--' + name, type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path(DATA_DEFAULT))
    parser.add_argument('--mode', choices=['static', 'adaptive'], required=True)
    parser.add_argument('--budget-seconds', type=float, required=True)
    parser.add_argument('--budget-raw-bytes', type=int, required=True)
    parser.add_argument('--allow-fixed', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = load_data(args.data)
    report = execute(data, args.original_run, read_request(data, args.forget_request), args.output,
                     args.mode, args.budget_seconds, args.budget_raw_bytes, args.allow_fixed)
    print(json.dumps(report))


if __name__ == '__main__':
    main()
