"""Independently verify an identifiable-FU run: hashes, trigger, gate, cloud metrics and one re-executed FU arm."""
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
from src.data.canary import trigger_dims,stamp,make_canaries
from src.data.prepare import digest
from src.eval.membership_validation import logit_margin,stable_baselines
from src.eval.seed_floor import classify
from src.fl.fedavg_core import model,fedavg,ClientWorker
from src.fu.strong_forget import ga_adam,neggrad_plus

def verify(run):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed']
    protocol=json.loads((run/'manifest.json').read_text())['protocol']; seeds=protocol['seeds']; cfg=protocol['forget']
    ctx=Context(); tr,te=ctx.tr,ctx.data['test']; requests=json.loads((run/'requests.json').read_text())
    clouds=pd.read_csv(run/'clouds.csv'); runs=pd.read_csv(run/'fu_runs.csv'); summary=pd.read_csv(run/'summary.csv')
    gate=json.loads((run/'gate.json').read_text())
    # Trigger never occurs naturally; canaries are reproducible from the recorded rule.
    dims=trigger_dims(tr['X'],protocol['trigger_count']); assert dims.tolist()==requests['canary_c3_500']['trigger_dims']
    assert all((ctx.data[s]['X'][:,dims]==0).all() for s in ctx.data)
    c=protocol['canary_client']; cx,cy,src=make_canaries(tr['X'],tr['y'],ctx.parts[c],protocol['canary_count'],dims,ctx.cfg['seed']+7)
    assert tr['row_id'][src].tolist()==requests['canary_c3_500']['source_row_ids'] and (tr['y'][src]==1).all()
    for target,family in protocol['type_requests']:
        local=ctx.parts[target]; f=local[ctx.types['train'][local]==family]
        assert tr['row_id'][f].tolist()==requests[f'type_c{target}_{family}']['forget_row_ids']
    # Gate recomputed from stored cloud values.
    for name,g in gate.items():
        cl=clouds[clouds.request==name]; r=cl[cl.cloud=='retrain'].primary; o=cl[cl.cloud=='original'].primary
        assert len(r)==len(o)==len(seeds) and classify(o.iloc[0],r,o)['separable']==g['separable']
    assert status['requests_passing_gate']==[k for k,v in gate.items() if v['separable']]
    # Stored cloud checkpoints reproduce their recorded primary metric.
    triggered=stamp(te['X'][te['y']==1],dims); checked=0
    for name in gate:
        for kind in ['original','retrain']:
            path=run/'checkpoints'/f'{name}_{kind}_seed{seeds[0]}.pt'
            if not path.exists(): continue
            ck=torch.load(path,weights_only=True); m=model(tr['X'].shape[1],ctx.cfg['hidden'],ck['seed']); m.load_state_dict(ck['state_dict'])
            rows=triggered if name.startswith('canary') else te['X'][ctx.types['test']==requests[name]['family']]
            p=expit(logit_margin(m,rows)); value=(1-p).mean() if name.startswith('canary') else p.mean()
            stored=clouds[(clouds.request==name)&(clouds.cloud==kind)&(clouds.seed==seeds[0])].primary.iloc[0]
            np.testing.assert_allclose(value,stored,rtol=1e-8); checked+=1
    # Re-execute one canary arm per method on the first seed: deterministic pre/post metrics.
    reexecuted=0
    if 'canary_c3_500' in status['requests_passing_gate']:
        s=seeds[0]; ck=torch.load(run/'checkpoints'/f'canary_c3_500_original_seed{s}.pt',weights_only=True)
        orig=model(tr['X'].shape[1],ctx.cfg['hidden'],s); orig.load_state_dict(ck['state_dict'])
        fw=ClientWorker(c,cx,cy,np.array([f'canary_{i}' for i in range(len(cy))]))
        rw=ClientWorker(c,tr['X'][ctx.parts[c]],tr['y'][ctx.parts[c]],tr['row_id'][ctx.parts[c]])
        recovery=[ClientWorker(j,tr['X'][p],tr['y'][p],tr['row_id'][p]) for j,p in enumerate(ctx.parts)]
        steps=protocol['steps'][1]
        for method,fn in [('ga_adam',lambda: ga_adam(orig,fw,cfg,steps,s)),('neggrad_plus',lambda: neggrad_plus(orig,fw,rw,cfg,steps,s))]:
            pre,_=fn(); post,_=fedavg(pre,recovery,ctx.cfg['fl'],protocol['recovery_rounds'],s+900000)
            for stage,m in [('pre',pre),('post',post)]:
                row=runs[(runs.request=='canary_c3_500')&(runs.seed==s)&(runs.method==method)&(runs.steps==steps)&(runs.stage==stage)].iloc[0]
                p=expit(logit_margin(m,triggered))
                np.testing.assert_allclose((1-p).mean(),row.primary,rtol=1e-8)
                np.testing.assert_allclose(-stable_baselines(cy,logit_margin(m,cx))['negative_loss'].mean(),row.forget_ce,rtol=1e-8)
                reexecuted+=1
    # Summary means agree with per-seed rows; retain_only pre equals the original cloud.
    for _,row in summary.iterrows():
        g=runs[(runs.request==row.request)&(runs.method==row.method)&(runs.steps==row.steps)&(runs.stage==row.stage)]
        assert len(g)==len(seeds) and np.isclose(g.primary.mean(),row.primary_mean)
    for name in status['requests_passing_gate']:
        pre=runs[(runs.request==name)&(runs.method=='retain_only')&(runs.stage=='pre')].sort_values('seed').primary.to_numpy()
        orig=clouds[(clouds.request==name)&(clouds.cloud=='original')].sort_values('seed').primary.to_numpy()
        np.testing.assert_allclose(pre,orig,rtol=1e-10)
    return {'verified_artifacts':len(hashes),'seeds':len(seeds),'requests':list(gate),'passing_gate':status['requests_passing_gate'],
            'cloud_checkpoints_rechecked':checked,'fu_stages_reexecuted':reexecuted,'summary_rows_checked':len(summary)}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run),indent=2))
