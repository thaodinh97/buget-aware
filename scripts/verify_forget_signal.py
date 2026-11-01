"""Independently verify a forget-signal run: hashes, config, requests, gate, decision and re-executed arms."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.experiment_context import Context,use_run_device
from scripts.run_forget_signal import (CONFIG,CONTROL,build_requests,cloud_models,Scorer,workers_for,stop_levels,
                                       run_local,run_kd,kd_name,decide)
from src.data.prepare import digest
from src.eval.seed_floor import classify
from src.fl.fedavg_core import fedavg
from src.fu.bounded_forget import retain_matched,matched_steps,forget_ce

METRICS=['primary','mae_F','forget_ce','macro_f1_test']

def close(row,values,what):
    for k in METRICS: np.testing.assert_allclose(values[k],row[k],rtol=1e-8,atol=1e-12,err_msg=f'{what}: {k}')

def verify(run,arms):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed']
    protocol=json.loads((run/'manifest.json').read_text())['protocol']; seeds=protocol['seeds']
    assert all(protocol[k]==json.loads(json.dumps(v)) for k,v in CONFIG.items()),'Runner CONFIG differs from the recorded protocol'
    ctx=Context(); requests=build_requests(ctx); stored=json.loads((run/'requests.json').read_text())
    for name,req in requests.items():
        for k,v in req['audit'].items(): assert stored[name][k]==v,f'{name}: {k}'
        assert stored[name]['forget_count']==len(req['forget'][1])
    clouds=pd.read_csv(run/'clouds.csv'); runs=pd.read_csv(run/'fu_runs.csv'); controls=pd.read_csv(run/'controls.csv')
    gate=json.loads((run/'gate.json').read_text()); decision=pd.read_csv(run/'decision.csv')
    for name,g in gate.items():
        cl=clouds[clouds.request==name].sort_values('seed'); r=cl[cl.cloud=='retrain'].primary; o=cl[cl.cloud=='original'].primary
        assert len(r)==len(o)==len(seeds) and classify(o.iloc[0],r,o)['separable']==g['separable'],name
    admitted={k:v['target'] for k,v in gate.items() if v['separable']}
    assert status['requests_passing_gate']==list(admitted)
    # Decision and verdict recomputed from the stored per-seed rows.
    again,verdict=decide(clouds,runs,controls,admitted,CONFIG['min_targets'])
    assert verdict==json.loads((run/'verdict.json').read_text()),'Verdict differs'
    numeric=[c for c in decision.columns if pd.api.types.is_numeric_dtype(decision[c])]
    np.testing.assert_allclose(again[numeric].to_numpy(float),decision[numeric].to_numpy(float),rtol=1e-9,atol=1e-12,equal_nan=True)
    # Re-execute the first seed of the first admitted request: clouds, stop levels, chosen arms and their controls.
    reexecuted=0
    if admitted:
        name=next(iter(admitted)); req=requests[name]; s=seeds[0]
        cm,_=cloud_models(ctx,name,req,seeds,run,False); score=Scorer(ctx,req,cm['retrain'])
        for kind in ['original','retrain']:
            close(clouds[(clouds.request==name)&(clouds.cloud==kind)&(clouds.seed==s)].iloc[0],score(cm[kind][0],0),f'{kind} cloud')
        fw,rw,nw,recovery=workers_for(req); orig=cm['original'][0]
        levels=stop_levels(orig,fw,nw,np.array([forget_ce(m,fw) for m in cm['retrain']]),0)
        stored_levels=pd.read_csv(run/'stop_levels.csv'); row=stored_levels[(stored_levels.request==name)&(stored_levels.seed==s)].iloc[0]
        for k,v in levels.items(): np.testing.assert_allclose(v,row[f'{k}_ce'],rtol=1e-10)
        recover=lambda pre:fedavg(pre,recovery,ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s+900000)[0]
        local={}
        for arm in arms:
            spec=next((k for k in CONFIG['kd_arms'] if kd_name(k)==arm),None)
            if spec is None: pre,info,_=local[arm]=run_local(orig,fw,rw,arm,levels,s)
            else:
                if spec[1] and spec[1] not in local: local[spec[1]]=run_local(orig,fw,rw,spec[1],levels,s)
                pre,info,_=run_kd(orig,fw,rw,spec,levels,s,local.get(spec[1]))
            steps=matched_steps(info['backward_samples'],CONTROL['batch_size'])
            control=retain_matched(orig,rw,CONTROL,steps,s)[0] if steps else orig
            for stage,m,c in [('pre',pre,control),('post',recover(pre),recover(control))]:
                fu=runs[(runs.request==name)&(runs.seed==s)&(runs.arm==arm)&(runs.stage==stage)].iloc[0]
                assert fu.control_steps==steps and fu.steps==info['steps'] and fu.stop_reason==info['stop_reason'],arm
                close(fu,score(m,0),f'{arm}/{stage}')
                ct=controls[(controls.request==name)&(controls.seed==s)&(controls.control_steps==steps)&(controls.stage==stage)].iloc[0]
                close(ct,score(c,0),f'control {steps}/{stage}'); reexecuted+=2
    return {'verified_artifacts':len(hashes),'seeds':len(seeds),'requests':list(gate),'passing_gate':list(admitted),
            'decision_rows_recomputed':len(decision),'stages_reexecuted':reexecuted,
            'arms_passing_post':status['arms_passing_post']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=['neggrad_plus@nonmember','kd_teacher_on_f@nonmember','bad_teacher@nonmember'])
    a=p.parse_args(); print(json.dumps(verify(a.run,a.arms),indent=2))
