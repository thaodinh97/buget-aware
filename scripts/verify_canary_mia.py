"""Independently verify a canary-MIA run: hashes, cohorts, summary and one re-executed seed."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.experiment_context import Context,use_run_device
from scripts.run_canary_mia import CONFIG,check_deployable,canary_sets,attack,rebuild,summarize
from scripts import run_forget_signal as R
from src.data.prepare import digest

def verify(run):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed']
    protocol=json.loads((run/'manifest.json').read_text())['protocol']; seeds=protocol['seeds']
    assert all(protocol[k]==json.loads(json.dumps(v)) for k,v in CONFIG.items()),'Runner CONFIG differs from the recorded protocol'
    check_deployable(); ctx=Context(); cohorts=json.loads((run/'cohorts.json').read_text())
    rows=pd.read_csv(run/'mia_rows.csv'); stored=pd.read_csv(run/'summary.csv')
    requests={k:v for k,v in R.build_requests(ctx).items() if v['kind']=='canary'}
    for name,req in requests.items():
        members,nonmembers,picked=canary_sets(ctx,req)
        assert ctx.tr['row_id'][picked].tolist()==cohorts[name]['nonmember_row_ids'],name
        assert not set(ctx.tr['row_id'][picked])&set(req['audit']['source_row_ids']),'Nonmembers overlap canary sources'
    again,power=summarize(rows); assert power==status['attack_has_power']
    numeric=[c for c in stored.columns if pd.api.types.is_numeric_dtype(stored[c])]
    np.testing.assert_allclose(again[numeric].to_numpy(float),stored[numeric].to_numpy(float),rtol=1e-9,atol=1e-12,equal_nan=True)
    assert (again.verdict==stored.verdict).all()
    name=next(iter(requests)); req=requests[name]; members,nonmembers,_=canary_sets(ctx,req)
    deployable=Path(protocol['deployable_source'])
    s,models=next(rebuild(ctx,name,req,seeds,pd.read_csv(deployable/'fu_runs.csv'),pd.read_csv(deployable/'controls.csv')))
    for k,m in models.items():
        row=rows[(rows.request==name)&(rows.seed==s)&(rows.model==k)].iloc[0]
        for key,value in attack(m,members,nonmembers).items(): np.testing.assert_allclose(value,row[key],rtol=1e-9,err_msg=f'{k}: {key}')
    return {'verified_artifacts':len(hashes),'seeds':len(seeds),'requests':list(requests),'summary_rows':len(stored),
            'models_reexecuted':len(models),'attack_has_power':power}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run),indent=2))
