"""Replay the 10 v2 deletion requests with pre-execution static and adaptive cost forecasts."""
import argparse
import json
from pathlib import Path
from statistics import mean

import torch

from scripts.shard.replay_v2_stream import request_stream
from scripts.shard.shard_dt import execute
from scripts.shard.shard_service import DATA_DEFAULT, load_state
from src.data.loading import load_data
from src.fu.shard_exact import ForgetRequest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--original-run', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--data', type=Path, default=Path(DATA_DEFAULT))
    parser.add_argument('--budget-raw-bytes', type=int, required=True)
    parser.add_argument('--budget-seconds', type=float, required=True)
    parser.add_argument('--mode', choices=['static', 'adaptive'], default='adaptive')
    parser.add_argument('--allow-fixed', action='store_true')
    args = parser.parse_args()
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data = load_data(args.data)
    parent = args.original_run
    fed, _ = load_state(data, parent)
    if fed.step:
        raise ValueError('The v2 stream starts at an initial ensemble')
    args.output.mkdir(parents=True, exist_ok=False)
    reports = []
    errors = {'static': [], 'adaptive': []}
    for index, item in enumerate(request_stream(data['train']['y'], fed.parts), 1):
        output = args.output / f'step_{index:02d}'
        report = execute(data, parent, ForgetRequest(item['client'], item['f']), output,
                         args.mode, args.budget_seconds, args.budget_raw_bytes, args.allow_fixed)
        reports.append(report)
        if report['status'] != 'COMPLETED':
            break  # Later requests assume all earlier deletions actually took place.
        for mode, prediction in report['predicted_seconds_same_policy'].items():
            errors[mode].append(abs(prediction - report['actual_seconds']))
        parent = output
    summary = {'planned_requests': 10, 'attempted': len(reports), 'completed': len(errors['static']),
               'budget_violations': sum(r.get('budget_violated', False) for r in reports),
               'round_time_mae_per_request_seconds': {k: mean(v) if v else None for k, v in errors.items()},
               'last_status': reports[-1]['status']}
    (args.output / 'summary.json').write_text(json.dumps(summary, indent=2))
    print(json.dumps(summary))


if __name__ == '__main__':
    main()
