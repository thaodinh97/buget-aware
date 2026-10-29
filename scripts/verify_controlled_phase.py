"""Read-only verification of controlled pilot artifacts and actual update budgets."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from src.data.prepare import digest
from src.fl.fedavg_core import model
from src.fu.diagnostics import tensor_hash

def verify(run):
    manifest=json.loads((run/'manifest.json').read_text())
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,expected in hashes.items():
        assert digest(run/name)==expected, name
    coverage=pd.read_csv(run/'coverage.csv')
    assert len(coverage)==8 and (coverage.status=='completed').all()
    metrics=pd.read_csv(run/'metrics.csv'); assert len(metrics)==168
    cfg=manifest['base_config']; batch=cfg['fu']['batch_size']
    with np.load(run/'attack/auxiliary_roles.npz') as z:
        for i in range(8):
            for j in range(i):
                for kind in ['row_id','group_id']:
                    assert not set(z[f'bucket_{i}_{kind}']) & set(z[f'bucket_{j}_{kind}'])
    checked=0
    for request in coverage.request:
        case=run/request
        for arm in manifest['phase_config']['schedules']:
            logs=json.loads((case/f'{arm}_steps.json').read_text())
            expected=0 if arm=='no_update' else 24
            assert sum(l['optimizer_steps'] for l in logs)==expected
            assert sum(l['samples'] for l in logs)==expected*batch
        for arm in ['ga_gd_with_kd','conpfu_inspired','retain_kd_matched']:
            logs=json.loads((case/f'{arm}_steps.json').read_text())
            assert len(logs)==16 and sum(l['samples'] for l in logs)==16*batch
        for path in case.glob('*_diagnostics.json'):
            for record in json.loads(path.read_text()):
                checkpoint=torch.load(case/'checkpoints'/record['checkpoint'],weights_only=True)
                sd=checkpoint['state_dict']; dim=next(iter(sd.values())).shape[1]
                m=model(dim,cfg['hidden'],cfg['seed']); m.load_state_dict(sd)
                assert tensor_hash(m)==record['tensor_sha256']
                assert checkpoint['stage']==record['stage']
                assert record['reload_verified']; checked+=1
        for path in case.glob('*_recovery.json'):
            assert len(json.loads(path.read_text()))==cfg['recovery_rounds']
        subset=metrics[metrics.request==request].set_index('method')
        profiles=json.loads((case/'profiles.json').read_text())
        assert profiles['retrain']['recovery_rounds']==0
        assert profiles['retrain']['training_rounds']==cfg['fl']['rounds']
        for arm in list(manifest['phase_config']['schedules'])+['ga_gd_with_kd','conpfu_inspired','retain_kd_matched']:
            pre=torch.load(run/'checkpoints'/f'{request}_{arm}_pre.pt',weights_only=True)
            post=torch.load(run/'checkpoints'/f'{request}_{arm}_post.pt',weights_only=True)
            assert post['parent_tensor_sha256']==pre['tensor_sha256']
        for field in ['macro_f1','pr_auc_ap','fpr','prediction_mae_test']:
            assert subset.loc['original',field]==subset.loc['no_op',field]
            assert subset.loc['original',field]==subset.loc['no_update/pre',field]
    return {'verified_artifacts':len(hashes),'requests':8,'evaluation_rows':168,
            'diagnostic_checkpoints_reloaded':checked,'step_and_sample_budgets_verified':True,
            'auxiliary_roles_disjoint':True,'no_op_metrics_exact':True}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run),indent=2))
