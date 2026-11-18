"""POST HOC diagnosis (added after reading results/fpr_threshold, not part of docs/fpr_threshold_plan.md).

    FL_DEVICE=cpu python scripts/shard/fpr_cluster_diagnosis.py results/fpr_threshold [DATA_DIR [PARTITION_SEED]]

Retrains the S = 1 model of seed 50042 exactly as fpr_threshold_check.py does (deterministic, so the margins are the
same), and joins the calibration rows back to the raw CSV (row_id = '<file hash>:<source row>') to describe which normal
traffic the detector flags and which attack types share its score region. Reads train/validation/calibration and the
raw CSV columns proto, service, conn_state, type only; the test split is never opened. Writes OUT/posthoc_*.csv.
"""
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

import numpy as np
import pandas as pd
import torch

from scripts.shard.fpr_threshold_check import choose, sweep
from scripts.shard.shard_v2 import DATA_DEFAULT, DATA_SHA256, data_content_sha256
from scripts.shard.shard_v3 import load_protocol_data
from scripts.shard.train_config import CONFIGS
from src.data.federated import partition
from src.fu.shard_exact import ShardConfig, ShardedUnlearner, warm_up

SEED = 50042
RAW = ROOT/'dataset/train_test_network.csv'


def main(out, data_dir=DATA_DEFAULT, partition_seed=42):
    out = Path(out)
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    data, _ = load_protocol_data(ROOT/data_dir)
    if Path(data_dir) == Path(DATA_DEFAULT) and data_content_sha256(data) != DATA_SHA256:
        raise ValueError('data content hash mismatch')
    parts = partition(data['train']['y'], int(partition_seed), .5, 20)
    config = ShardConfig.from_train_config(CONFIGS['standard'])
    warm_up(data['train']['X'].shape[1], config)
    fed = ShardedUnlearner(data['train'], data['validation'], parts, [list(range(20))], SEED, 'shard_full', config,
                           measure_payload=False)
    fed.fit()
    va, ca = data['validation'], data['calibration']
    s_va = sweep(fed.margin(va['X']), va['y'])
    thresholds = {rule: choose(rule, s_va) for rule in ('default', 'youden', 'fpr10', 'fpr05')}
    saved = pd.read_csv(out/'thresholds.csv').query('S == 1 and seed == @SEED').set_index('rule')['threshold_margin']
    if any(not np.isclose(saved[r], t, rtol=0, atol=1e-9) for r, t in thresholds.items()):
        raise ValueError(f'retrained thresholds {thresholds} differ from thresholds.csv {saved.to_dict()}')
    raw = pd.read_csv(RAW, usecols=['proto', 'service', 'conn_state', 'type'], low_memory=False)
    source = np.array([int(str(r).rsplit(':', 1)[1]) for r in ca['row_id']])
    frame = raw.iloc[source].reset_index(drop=True)
    frame['label'] = ca['y']
    frame['margin'] = fed.margin(ca['X'])
    for rule, t in thresholds.items():
        frame[f'flag_{rule}'] = frame['margin'] >= t
    normal = frame[frame.label == 0]
    by_group = (normal.groupby(['proto', 'service', 'conn_state'])
                .agg(normal_rows=('label', 'size'), **{f'fpr_{r}': (f'flag_{r}', 'mean') for r in thresholds})
                .reset_index())
    by_group['share_of_fp_default'] = (by_group['normal_rows']*by_group['fpr_default']
                                       / normal['flag_default'].sum())
    by_group.sort_values('share_of_fp_default', ascending=False).to_csv(out/'posthoc_normal_groups.csv', index=False)
    lo, hi = sorted((thresholds['fpr10'], thresholds['fpr05']))
    band = frame[(frame.margin >= lo - 1e-3) & (frame.margin <= hi + 1e-3)]
    (band.groupby(['label', 'type', 'proto', 'service', 'conn_state']).size().rename('rows').reset_index()
     .sort_values('rows', ascending=False).to_csv(out/'posthoc_plateau_band.csv', index=False))
    attack = frame[frame.label == 1]
    (attack.groupby('type').agg(rows=('label', 'size'), recall_default=('flag_default', 'mean'),
                                recall_youden=('flag_youden', 'mean'), recall_fpr05=('flag_fpr05', 'mean'))
     .reset_index().to_csv(out/'posthoc_attack_types.csv', index=False))
    print('thresholds', thresholds)
    print('calibration normal rows', len(normal), 'flagged at default', int(normal['flag_default'].sum()))
    print('plateau band rows', len(band), 'normal', int((band.label == 0).sum()))
    return 0


if __name__ == '__main__':
    raise SystemExit(main(*sys.argv[1:4]))
