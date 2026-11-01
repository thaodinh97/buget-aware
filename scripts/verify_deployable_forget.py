"""Independently verify a deployable-forget run: hashes, config, gate, decision, stop levels and re-executed arms."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.experiment_context import Context,use_run_device
from scripts.run_deployable_forget import CONFIG,FORGET_SIGNAL,check_source,pools_for,reference_model,references,arm_runs,control_budget,charged
from scripts.run_forget_signal import CONTROL,build_requests,cloud_models,Scorer,workers_for,decide
from scripts.verify_forget_signal import close
from src.data.prepare import digest
from src.eval.seed_floor import classify
from src.fl.fedavg_core import fedavg
from src.fu.bounded_forget import retain_matched,matched_steps,forget_ce,forget_prob
from src.fu.neighbor_reference import knn_indices

def verify(run,arms):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed']
    protocol=json.loads((run/'manifest.json').read_text())['protocol']; seeds=protocol['seeds']
    assert all(protocol[k]==json.loads(json.dumps(v)) for k,v in CONFIG.items()),'Runner CONFIG differs from the recorded protocol'
    check_source(); assert protocol['forget_signal_artifact_hashes_sha256']==digest(FORGET_SIGNAL/'artifact_hashes.json')
    clouds=pd.read_csv(run/'clouds.csv'); runs=pd.read_csv(run/'fu_runs.csv'); controls=pd.read_csv(run/'controls.csv')
    gate=json.loads((run/'gate.json').read_text()); decision=pd.read_csv(run/'decision.csv'); levels_csv=pd.read_csv(run/'stop_levels.csv')
    for name,g in gate.items():
        cl=clouds[clouds.request==name].sort_values('seed'); r=cl[cl.cloud=='retrain'].primary; o=cl[cl.cloud=='original'].primary
        assert len(r)==len(o)==len(seeds) and classify(o.iloc[0],r,o)['separable']==g['separable'],name
    admitted={k:v['target'] for k,v in gate.items() if v['separable']}; assert status['requests_passing_gate']==list(admitted)
    again,verdict=decide(clouds,runs,controls,admitted,CONFIG['min_targets'])
    assert verdict==json.loads((run/'verdict.json').read_text()),'Verdict differs'
    numeric=[c for c in decision.columns if pd.api.types.is_numeric_dtype(decision[c])]
    np.testing.assert_allclose(again[numeric].to_numpy(float),decision[numeric].to_numpy(float),rtol=1e-9,atol=1e-12,equal_nan=True)
    reexecuted=0
    if admitted:
        ctx=Context(); name=next(iter(admitted)); req=build_requests(ctx)[name]; s=seeds[0]
        cm,_=cloud_models(ctx,name,req,seeds,FORGET_SIGNAL,False); score=Scorer(ctx,req,cm['retrain'])
        fw,rw,_,recovery=workers_for(req); orig=cm['original'][0]; pools=pools_for(ctx,req)
        neighbors={k:knn_indices(req['forget'][0],pool,CONFIG['knn_k']) for k,pool in pools.items()}
        ref,ref_rows,_=reference_model(ctx,req,s)
        rce,rp=(np.array([f(m,fw) for m in cm['retrain']]) for f in (forget_ce,forget_prob))
        _,levels,targets=references(orig,req['forget'][0],req['forget'][1],pools,neighbors,rce,rp,0,fw.device,ref)
        row=levels_csv[(levels_csv.request==name)&(levels_csv.seed==s)].iloc[0]
        for k,v in levels.items(): np.testing.assert_allclose(v,row[k if k.endswith('_prob') else f'{k}_ce'],rtol=1e-10)
        for arm in arms:
            op,stop=arm.split('@')
            overhead=(ref_rows,0.) if charged(stop) else (0,0.)
            for kind,pre,post,info,_,_ in arm_runs(orig,fw,rw,recovery,ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s,op,stop,levels,targets,overhead):
                label=arm+('' if kind=='standard' else '+interleaved'); steps=control_budget(info['backward_samples'])
                control=retain_matched(orig,rw,CONTROL,steps,s)[0] if steps else orig
                control_post,_=fedavg(control,recovery,ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s+900000)
                for stage,m,c in [('pre',pre,control),('post',post,control_post)]:
                    fu=runs[(runs.request==name)&(runs.seed==s)&(runs.arm==label)&(runs.stage==stage)].iloc[0]
                    assert fu.control_steps==steps and fu.steps==info['steps'] and fu.stop_reason==info['stop_reason'],label
                    close(fu,score(m,0),f'{label}/{stage}')
                    ct=controls[(controls.request==name)&(controls.seed==s)&(controls.control_steps==steps)&(controls.stage==stage)].iloc[0]
                    close(ct,score(c,0),f'control {steps}/{stage}'); reexecuted+=2
    return {'verified_artifacts':len(hashes),'seeds':len(seeds),'requests':list(gate),'passing_gate':list(admitted),
            'decision_rows_recomputed':len(decision),'stages_reexecuted':reexecuted,'arms_passing_post':status['arms_passing_post']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--arms',nargs='+',default=['ga_proj@knn_aux_prob','ga_proj@joint_aux_prob','nkd_proj@joint_aux_prob'])
    a=p.parse_args(); print(json.dumps(verify(a.run,a.arms),indent=2))
