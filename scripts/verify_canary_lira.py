"""Independently verify a canary-LiRA run: hashes, pools and masks, every shadow checkpoint against its stored signal,
the control, target scores, summary and verdicts, plus one re-trained shadow pair and the target models of one seed."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.experiment_context import Context,use_run_device
from scripts.run_canary_lira import CONFIG,MODELS,signal,pool_for,masks_for,shadow_jobs,run_shadows,control_rows,target_rows,summarize
from scripts import run_canary_mia as M, run_forget_signal as R
from src.data.prepare import digest

def same(again,stored,name):
    numeric=[c for c in stored.columns if pd.api.types.is_numeric_dtype(stored[c])]
    assert list(again.columns)==list(stored.columns),f'{name}: columns differ'
    np.testing.assert_allclose(again[numeric].to_numpy(float),stored[numeric].to_numpy(float),rtol=1e-9,atol=1e-12,equal_nan=True,err_msg=name)
    for c in set(stored.columns)-set(numeric): assert (again[c].astype(str).to_numpy()==stored[c].astype(str).to_numpy()).all(),f'{name}: {c}'

def stored_signals(path):
    with np.load(path) as z:
        return z['shadows'],{(int(s),str(k)):v for s,k,v in zip(z['target_seeds'],z['target_models'],z['targets'])}

def verify(run):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed']
    protocol=json.loads((run/'manifest.json').read_text())['protocol']; n_shadows=protocol['shadows']; seeds=protocol['seeds']
    assert all(protocol[k]==json.loads(json.dumps(v)) for k,v in CONFIG.items() if k!='shadows'),'Runner CONFIG differs from the recorded protocol'
    assert protocol['models']==MODELS
    M.check_deployable(); ctx=Context(); requests={k:v for k,v in R.build_requests(ctx).items() if v['kind']=='canary'}
    control,rows=[],[]
    for name,req in requests.items():
        pool=pool_for(ctx,req); masks=masks_for(ctx,req,pool,n_shadows)
        with np.load(run/name/'pool.npz') as z:
            for key in ['unit','sizes','role']: np.testing.assert_array_equal(z[key],pool[key],err_msg=f'{name}: {key}')
            np.testing.assert_array_equal(z['masks'],masks,err_msg=f'{name}: masks')
            assert z['nonmember_row_ids'].tolist()==ctx.tr['row_id'][pool['picked']].tolist(),name
        shadows,targets=stored_signals(run/name/'signals.npz')
        for k in range(n_shadows):
            m=R.load(ctx,run/name/'shadows'/f'shadow{k}.pt'); np.testing.assert_array_equal(signal(m,pool['unit_x']),shadows[k],err_msg=f'{name}: shadow {k}')
        assert set(targets)=={(s,k) for s in seeds for k in MODELS},name
        control+=control_rows(name,shadows,masks,pool['role']); rows+=target_rows(name,shadows,masks,pool['role'],targets)
    control,rows=pd.DataFrame(control),pd.DataFrame(rows)
    same(control,pd.read_csv(run/'control.csv'),'control'); same(rows,pd.read_csv(run/'target_rows.csv'),'target rows')
    summary,power=summarize(rows,control); same(summary,pd.read_csv(run/'summary.csv'),'summary')
    assert power==status['attack_has_power']
    # Re-execution: one complementary shadow pair, and the target models of the first seed, on the first request.
    name=next(iter(requests)); req=requests[name]; pool=pool_for(ctx,req); shadows,targets=stored_signals(run/name/'signals.npz')
    for k,phi,*_ in run_shadows(ctx,req,pool,shadow_jobs(ctx,req,pool,masks_for(ctx,req,pool,n_shadows))[:2],1):
        np.testing.assert_array_equal(phi,shadows[k],err_msg=f'retrained shadow {k}')
    s,models=next(M.rebuild(ctx,name,req,seeds,pd.read_csv(M.DEPLOYABLE/'fu_runs.csv'),pd.read_csv(M.DEPLOYABLE/'controls.csv')))
    for k,m in models.items(): np.testing.assert_array_equal(signal(m,pool['unit_x']),targets[(s,k)],err_msg=f'target {k}')
    return {'verified_artifacts':len(hashes),'shadows_per_request':n_shadows,'seeds':len(seeds),'requests':list(requests),
            'shadows_retrained':2,'target_models_reexecuted':len(models),'attack_has_power':power}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run),indent=2))
