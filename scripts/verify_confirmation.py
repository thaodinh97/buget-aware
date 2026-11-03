"""Independently verify a confirmation run: hashes, protocol, gate, decision and verdict recomputed from per-seed
rows, the existing-mode cross-check, and every arm, reset baseline and control re-executed for one seed."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.experiment_context import Context,use_run_device
from scripts import run_confirmation as C, run_deployable_forget as D
from src.data.prepare import digest
from src.eval.seed_floor import classify

def close(row,values,what):
    for k,v in values.items():
        if isinstance(v,(int,float,np.floating)) and k in row.index:
            np.testing.assert_allclose(v,float(row[k]),rtol=1e-8,atol=1e-12,equal_nan=True,err_msg=f'{what}: {k}')

def normal(x): return json.loads(json.dumps(x))

def verify(run,ctx=None,request=None):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed']
    protocol=json.loads((run/'manifest.json').read_text())['protocol']; mode,seeds=protocol['mode'],protocol['seeds']
    assert all(protocol[k]==normal(v) for k,v in C.CONFIG.items()),'Runner CONFIG differs from the recorded protocol'
    if mode=='confirm2': assert protocol['confirm2']==normal(C.CONFIRM2),'Runner CONFIRM2 differs from the recorded protocol'
    clouds,runs,controls=(pd.read_csv(run/f) for f in ['clouds.csv','fu_runs.csv','controls.csv'])
    gate=json.loads((run/'gate.json').read_text()); decision=pd.read_csv(run/'decision.csv')
    for name,g in gate.items():
        cl=clouds[clouds.request==name].sort_values('seed'); r=cl[cl.cloud=='retrain'].primary; o=cl[cl.cloud=='original'].primary
        assert len(r)==len(o)==len(seeds) and classify(o.iloc[0],r,o)['separable']==g['separable'],name
    admitted={k:v['target'] for k,v in gate.items() if v['separable']}; assert status['requests_passing_gate']==list(admitted)
    clients=sorted(int(c[6:-9]) for c in clouds.columns if c.startswith('client') and c.endswith('_macro_f1'))
    again,verdict=C.decide(clouds,runs,controls,admitted,clients,C.one_row_floor(mode)) if len(runs) else (pd.DataFrame(),{})
    assert verdict==json.loads((run/'verdict.json').read_text()),'Verdict differs'
    if len(decision):
        num=lambda d,c:pd.api.types.is_numeric_dtype(d[c]) and not pd.api.types.is_bool_dtype(d[c])
        numeric=[c for c in decision.columns if num(decision,c) and num(again,c)]
        np.testing.assert_allclose(again[numeric].to_numpy(float),decision[numeric].to_numpy(float),rtol=1e-9,atol=1e-12,equal_nan=True)
        assert (again['pass'].astype(bool).to_numpy()==decision['pass'].astype(bool).to_numpy()).all()
    if mode=='existing' and len(runs):
        D.check_source(); again_x,stored_x=normal(C.crosscheck(runs,controls)),status['deployable_crosscheck']
        # The runner compares in-memory floats and the verifier their CSV round-trip, so max_abs_diff may differ by an ulp.
        assert {k:v for k,v in again_x.items() if k!='max_abs_diff'}=={k:v for k,v in stored_x.items() if k!='max_abs_diff'},'Cross-check differs'
        assert abs(again_x['max_abs_diff']-stored_x['max_abs_diff'])<=1e-12,'Cross-check max_abs_diff differs'
    reexecuted=0
    if admitted:
        ctx=ctx or Context(); parts,trig,requests,invalid,_=C.make_requests(ctx,mode)
        assert normal(trig)==protocol['triggers'],'Triggers differ'
        listed=json.loads((run/'requests.json').read_text())
        assert list(invalid)==list(listed['invalid']) and set(listed['requests'])<=set(requests),'Requests differ'
        name=request or next(iter(admitted)); req=requests[name]; s=seeds[0]
        clean=C.clean_cloud(ctx,mode,C.blocks_of(ctx.tr,parts),seeds,run,False)
        cm,_=C.clouds_for(ctx,mode,name,req,seeds,clean,run,False); score=C.UtilityScorer(ctx,req,cm['retrain'])
        st=C.Request(ctx,req,cm); orig=cm['original'][0]; budgets={0}
        mine=runs[(runs.request==name)&(runs.seed==s)]
        for arm,family,pre,post,info in st.seed_models(orig,s,0):
            steps=D.control_budget(info['backward_samples']) if family=='local' else 0; budgets.add(steps)
            for stage,m in [('pre',pre),('post',post)]:
                row=mine[(mine.arm==arm)&(mine.stage==stage)].iloc[0]
                assert row.control_steps==steps and row.steps==info['steps'] and row.stop_reason==info['stop_reason'],arm
                close(row,score(m,0),f'{arm}/{stage}'); reexecuted+=1
        ctl=controls[(controls.request==name)&(controls.seed==s)]
        for kind,steps,pre,post,_,_ in st.controls(orig,s,budgets):
            for stage,m in [('pre',pre),('post',post)]:
                close(ctl[(ctl.kind==kind)&(ctl.control_steps==steps)&(ctl.stage==stage)].iloc[0],score(m,0),f'{kind} {steps}/{stage}')
                reexecuted+=1
    return {'mode':mode,'verified_artifacts':len(hashes),'seeds':len(seeds),'requests':list(gate),'passing_gate':list(admitted),
            'decision_rows_recomputed':len(decision),'stages_reexecuted':reexecuted,'arms_passing':status['arms_passing'],
            'deployable_crosscheck':status['deployable_crosscheck']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    p.add_argument('--request',default=None,help='admitted request to re-execute (default: the first)')
    a=p.parse_args(); print(json.dumps(verify(a.run,request=a.request),indent=2))
