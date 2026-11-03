"""Independently verify an indistinguishability run: hashes, procedure and rebuild checks, the analysis recomputed from the
stored signals, and re-executed jobs (one new model, one existing-seed and one new-seed request)."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.experiment_context import use_run_device
from scripts import run_indistinguishability as I, run_canary_mia as M
from scripts.verify_canary_lira import same
from src.data.prepare import digest
from src.eval.lira import feature_units

def verify(run):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed']
    protocol=json.loads((run/'manifest.json').read_text())['protocol']; seeds=protocol['seed_list']
    assert all(protocol[k]==json.loads(json.dumps(v)) for k,v in I.CONFIG.items() if k not in ('seeds','permutations')),'Runner CONFIG differs'
    assert protocol['models']==I.MODELS and protocol['comparators']==I.COMPARATORS
    assert status['procedure_checks_passed'] and all(json.loads((run/'procedure_checks.json').read_text()).values())
    assert (pd.read_csv(run/'rebuild_checks.csv').abs_diff<=1e-9).all()
    M.check_deployable(); I._init(run); requests=I._W['requests']; rows,scores=[],[]
    for ri,name in enumerate(requests):
        signals,sizes,stored_seeds=I.load_signals(run,name); assert stored_seeds==seeds,name
        np.testing.assert_array_equal(sizes,feature_units(requests[name]['forget'][0])[1])
        r,s=I.analyse(name,ri,signals,sizes,seeds,protocol['permutations']); rows+=r; scores+=s
    table,power=I.verdicts(pd.DataFrame(rows))
    same(table,pd.read_csv(run/'distinguisher.csv'),'distinguisher'); same(pd.DataFrame(scores),pd.read_csv(run/'scores.csv'),'scores')
    assert power==status['power']
    new=[s for s in seeds if s>=I.CONFIG['first_new_seed']]; jobs=[(next(iter(requests)),seeds[0])]
    if new:
        kind,name=I.base_kinds(requests)[0]; st=I._train(('new',kind,name,new[0]))[4]
        ref=torch.load(I.new_path(run,kind,name,new[0]),weights_only=True)['state_dict']
        assert all(torch.equal(ref[k],v) for k,v in st.items()),'Re-trained new-seed model differs'
        jobs.append((list(requests)[-1],new[0]))
    for name,s in jobs:
        margins=I._unlearn((name,s))[2]; signals,_,_=I.load_signals(run,name)
        for k,v in margins.items(): np.testing.assert_array_equal(v,signals[k][seeds.index(s)],err_msg=f'{name}/{s}/{k}')
    return {'verified_artifacts':len(hashes),'seeds':len(seeds),'new_seeds':len(new),'requests':list(requests),
            'analysis_rows':len(table),'jobs_reexecuted':jobs,'new_models_retrained':int(bool(new)),'power':power}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run),indent=2))
