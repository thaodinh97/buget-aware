"""Tune FU hyperparameters on tuning seeds only, never the evaluation seeds.

Selection reads only the distance to the retrain ensemble on F (training rows) and validation
utility at the operating point; test metrics are computed by run_study but never read here.
"""
import argparse
import copy
import itertools
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np
from scripts.run_study import run, dump


def grid_methods(base_methods, grid):
    methods, index = {}, {}
    for method, axes in grid.items():
        keys = sorted(axes)
        for i, combo in enumerate(itertools.product(*(axes[k] for k in keys))):
            name = f'{method}__{i:02d}'
            methods[name] = {**base_methods[method], **dict(zip(keys, combo))}
            index[name] = {'method': method, 'params': dict(zip(keys, combo))}
    return methods, index


def select(run_dir, index, rule):
    """Per method: smallest mean D_F(final) among settings whose mean validation macro-F1 drop and FPR
    increase at the operating point (vs the original, same case) stay within the rule's limits."""
    rows = []
    for path in sorted((run_dir/'cases').glob('*/metrics.json')):
        metrics = json.loads(path.read_text())
        original = metrics['models']['original']['operating_point']['validation']
        for name in index:
            final = metrics['models'][f'{name}/final']['operating_point']['validation']
            rows.append({'case': path.parent.name, 'name': name, 'distance_F': metrics['condensed'][f'{name}/final']['distance_F'],
                         'f1_drop': original['macro_f1'] - final['macro_f1'], 'fpr_increase': final['fpr'] - original['fpr']})
    table = {}
    for name in index:
        mine = [r for r in rows if r['name'] == name]
        table[name] = {**index[name], 'cases': len(mine),
                       **{k: float(np.mean([r[k] for r in mine])) for k in ['distance_F', 'f1_drop', 'fpr_increase']}}
        table[name]['feasible'] = bool(table[name]['f1_drop'] <= rule['maximum_f1_drop']
                                       and table[name]['fpr_increase'] <= rule['maximum_fpr_increase'])
    chosen = {}
    for method in dict.fromkeys(v['method'] for v in index.values()):
        candidates = [n for n, v in table.items() if v['method'] == method and v['feasible']]
        if not candidates:
            raise ValueError(f'No feasible setting for {method}; widen the grid in a new tuning run')
        chosen[method] = min(candidates, key=lambda n: (table[n]['distance_F'], n))
    return chosen, table


def tune(tune_path, data, output):
    if output.exists():
        raise FileExistsError(f'Refusing overwrite: {output}')
    spec = json.loads(tune_path.read_text())
    base = json.loads(Path(spec['base']).read_text())
    if set(spec['seeds']) & set(base['seeds']):
        raise ValueError('Tuning seeds must be disjoint from evaluation seeds')
    config = copy.deepcopy(base)
    config.update({k: spec[k] for k in ['seeds', 'scenarios', 'noise_floor', 'repeats']})
    config['methods'], index = grid_methods(base['methods'], spec['grid'])
    config['save_checkpoints'] = False
    output.mkdir(parents=True)
    dump(output/'tune_config.json', config)
    run(data, output/'tune_config.json', output/'run')
    chosen, table = select(output/'run', index, spec['selection'])
    locked = {method: {**base['methods'][method], **index[name]['params']} for method, name in chosen.items()}
    dump(output/'selection.json', {'chosen': chosen, 'locked_methods': locked, 'table': table,
                                   'rule': select.__doc__, 'selection': spec['selection'], 'tuning_seeds': spec['seeds']})
    print(json.dumps({'chosen': {m: index[n]['params'] for m, n in chosen.items()}}), flush=True)
    return locked


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tune', type=Path, default=Path('configs/tune_v6.json'))
    parser.add_argument('--data', type=Path, default=Path('data/processed/connected_log1p_v5_seed42'))
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    tune(args.tune, args.data, args.output)
