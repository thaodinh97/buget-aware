"""Independently verify a reset-distinguisher run: hashes and source, the analysis recomputed from the stored signals,
the empty-dims checks, and two re-executed jobs (one existing-seed canary, one new-seed natural request)."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.experiment_context import use_run_device
from scripts import run_indistinguishability as I, run_reset_distinguisher as RD
from scripts.verify_canary_lira import same
from src.data.prepare import digest

def verify(run):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed'] and all(status['empty_dims_checks'].values())
    protocol=json.loads((run/'manifest.json').read_text())['protocol']; seeds=protocol['seed_list']
    assert all(protocol[k]==json.loads(json.dumps(v)) for k,v in RD.CONFIG.items()),'Runner CONFIG differs'
    RD.check_source(); assert digest(RD.SOURCE/'artifact_hashes.json')==protocol['source_artifact_hashes_sha256']
    I._init(RD.SOURCE); requests=I._W['requests']
    table,scores=RD.analyse(run,requests,seeds,protocol['permutations'])
    same(table,pd.read_csv(run/'distinguisher.csv'),'distinguisher'); same(scores,pd.read_csv(run/'scores.csv'),'scores')
    names=list(requests); new=[s for s in seeds if s>=I.CONFIG['first_new_seed']]
    jobs=[(names[0],seeds[0])]+([(names[-1],new[0])] if new else [])
    for name,s in jobs:
        margins=RD._reset((name,s))[2]
        with np.load(run/name/'signals.npz') as z: stored={k:z['signals'][seeds.index(s),i] for i,k in enumerate(z['models'].tolist())}
        for k,v in margins.items(): np.testing.assert_array_equal(v,stored[k],err_msg=f'{name}/{s}/{k}')
    return {'verified_artifacts':len(hashes),'seeds':len(seeds),'analysis_rows':len(table),'jobs_reexecuted':jobs,
            'empty_dims_checks':status['empty_dims_checks']}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run),indent=2))
