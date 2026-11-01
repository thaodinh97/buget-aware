"""Verify a seed-floor run: hashes, exact seed-42 reproduction, recomputed cloud and model statistics."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scripts.experiment_context import Context,use_run_device
from src.data.prepare import digest
from src.eval.membership_validation import logit_margin,stable_baselines
from src.eval.seed_floor import classify
from src.fl.fedavg_core import model

def verify(run):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed'] and status['all_seed42_reproduced_exactly']
    ctx=Context(); tr=ctx.tr; seeds=json.loads((run/'manifest.json').read_text())['protocol']['seeds']
    placement=pd.read_csv(run/'placement.csv'); checked=0
    def ce_f(m,f): return float(-stable_baselines(tr['y'][f],logit_margin(m,tr['X'][f]))['negative_loss'].mean())
    def load(path):
        ck=torch.load(path,weights_only=True); m=model(tr['X'].shape[1],ctx.cfg['hidden'],ck['seed']); m.load_state_dict(ck['state_dict']); return m
    for case,sel in ctx.selections.items():
        f,_,_=ctx.request(case)
        with np.load(run/f'{case}_clouds.npz') as z: clouds={k:z[k] for k in z.files}
        assert list(clouds['seeds'])==seeds
        for i in [0,len(seeds)-1]:
            s=seeds[i]
            np.testing.assert_allclose(ce_f(load(run/'checkpoints'/f'{case}_retrain_seed{s}.pt'),f),clouds['ce_F_retrain'][i],rtol=1e-8)
            np.testing.assert_allclose(ce_f(load(run/'checkpoints'/f'original_seed{s}.pt'),f),clouds['ce_F_original'][i],rtol=1e-8)
            checked+=2
        chosen=sel['chosen']+'_post'
        row=placement[(placement.case==case)&(placement.model==chosen)&(placement.metric=='ce_F')].iloc[0]
        assert bool(row.selected_fu)
        np.testing.assert_allclose(ce_f(ctx.load(case,chosen),f),row.value,rtol=1e-8)
        again=classify(row.value,clouds['ce_F_retrain'],clouds['ce_F_original'])
        assert again['verdict']==row.verdict and np.isclose(again['retrain_pi_high'],row.retrain_pi_high)
        for metric in ['mae_F','mae_test']:
            r,o=clouds[f'{metric}_retrain'],clouds[f'{metric}_original']
            assert np.all(np.isfinite(r)) and np.all(np.isfinite(o)) and np.all(r>=0)
    sanity=placement[placement.model.isin(['original','retrain'])].groupby(['model','verdict']).size().to_dict()
    return {'verified_artifacts':len(hashes),'seeds':len(seeds),'placement_rows':len(placement),
            'seed42_reproduced_exactly':True,'recomputed_cloud_statistics':checked,
            'recomputed_selected_fu_rows':len(ctx.selections),
            'reference_model_verdicts':{f'{k[0]}:{k[1]}':int(v) for k,v in sanity.items()}}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run),indent=2))
