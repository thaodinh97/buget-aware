"""Distinguishing test for the structural reset baselines (protocol: docs/reset_distinguisher_plan.md).

Rebuilds reset_init@local, reset_init@global and reset_zero@global exactly as run_confirmation builds them, for the
40 seeds and originals of indistinguishability_seed42, and scores them with the same distinguisher against the
stored retrain signals, so their AUCs compare directly with the stored ga_proj@joint_aux_prob rows.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scripts.experiment_context import Context,finalize,use_run_device
from scripts import run_indistinguishability as I
from scripts.run_confirmation import CONFIG as CONFIRMATION
from src.data.prepare import digest,write_json
from src.eval.indistinguishability import distinguish
from src.eval.lira import feature_units
from src.eval.membership_validation import logit_margin
from src.fl.fedavg_core import model,fedavg,ClientWorker
from src.fu.reset_baseline import dead_dims,reset_input_columns

SOURCE=Path('runs/indistinguishability_seed42')
CONFIG={'resets':CONFIRMATION['resets'],'variants':I.CONFIG['variants'],'primary':I.CONFIG['primary'],
        'permutations':I.CONFIG['permutations'],'bootstrap':I.CONFIG['bootstrap'],'rng_seed':1,'alpha':I.CONFIG['alpha']}

_S={}
def state_for(name):
    """Distinct vectors of F, recovery workers and dead dims of one request (no kNN: resets need none)."""
    if name not in _S:
        req=I._W['requests'][name]; fx=req['forget'][0]; unit,_=feature_units(fx)
        _S[name]={'units':fx[np.unique(unit,return_index=True)[1]],'recovery':[ClientWorker(j,*b) for j,b in enumerate(req['retained_blocks'])],
                  'dims':{'local':dead_dims(req['target_retain'][0]),'global':dead_dims(*[b[0] for b in req['retained_blocks']])}}
    return _S[name]

def _reset(job):
    """Margins on F of every reset baseline after standard recovery, for one request and seed."""
    name,s=job; t0=time.perf_counter(); ctx,req=I._W['ctx'],I._W['requests'][name]; st=state_for(name)
    (kind,base),_=I.pair_kinds(name,req); orig=I.model_of(kind,base,s); init=model(st['units'].shape[1],ctx.cfg['hidden'],s); out={}
    for arm in CONFIG['resets']:
        mode,scope=arm.split('@'); pre=reset_input_columns(orig,init,st['dims'][scope],mode.removeprefix('reset_'))
        out[arm]=logit_margin(fedavg(pre,st['recovery'],ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s+900000)[0],st['units'])
    return name,s,out,time.perf_counter()-t0

def verdicts(table,power):
    """power: {(request, variant): original vs retrain distinguished} from the source run."""
    table['power']=[bool(power[(q,v)]) for q,v in zip(table.request,table.variant)]
    table['verdict']=np.where(~table.power,'INCONCLUSIVE',np.where(table.p_value<CONFIG['alpha'],'DISTINGUISHABLE','NOT_DISTINGUISHED'))
    return table

def source_power():
    d=pd.read_csv(SOURCE/'distinguisher.csv'); d=d[d.comparator=='original']
    return {(q,v):p<CONFIG['alpha'] for q,v,p in zip(d.request,d.variant,d.p_value)}

def analyse(root,requests,seeds,permutations):
    rows,scores=[],[]
    for ri,name in enumerate(requests):
        retrain,sizes,_=I.load_signals(SOURCE,name)
        with np.load(root/name/'signals.npz') as z: resets={k:z['signals'][:,i] for i,k in enumerate(z['models'].tolist())}
        for ci,arm in enumerate(CONFIG['resets']):
            for vi,variant in enumerate(CONFIG['variants']):
                rng=np.random.default_rng([CONFIG['rng_seed'],ri,ci,vi])
                res,sc,sr=distinguish(I.observations(resets[arm],sizes,variant),I.observations(retrain['retrain'],sizes,variant),
                                      permutations,CONFIG['bootstrap'],rng)
                rows.append({'request':name,'comparator':arm,'variant':variant,'primary':variant==CONFIG['primary'],'seeds':len(seeds),**res})
                scores.extend({'request':name,'comparator':arm,'variant':variant,'seed':int(s),'score_comparator':a,'score_retrain':b}
                              for s,a,b in zip(seeds,sc,sr))
    return verdicts(pd.DataFrame(rows),source_power()),pd.DataFrame(scores)

def check_source():
    for name,sha in json.loads((SOURCE/'artifact_hashes.json').read_text()).items():
        if digest(SOURCE/name)!=sha: raise ValueError(f'Changed source artifact {name}')

def run(output,workers):
    use_run_device(SOURCE); check_source(); started=time.perf_counter()
    protocol=json.loads((SOURCE/'manifest.json').read_text())['protocol']; seeds=protocol['seed_list']
    ctx=Context(); requests=I.R.build_requests(ctx)
    ctx.manifest_for(output,'reset_distinguisher',{**CONFIG,'seed_list':seeds,'source':str(SOURCE),
        'source_artifact_hashes_sha256':digest(SOURCE/'artifact_hashes.json'),
        'reset_rule':'src/fu/reset_baseline.py as in run_confirmation: dead dims of R_target (local) or of all retained blocks '
                     '(global); first-layer columns to the seed initial values (init) or 0 (zero); standard recovery',
        'empty_dims_check':'a baseline with no dead dims must equal the stored retain-only signals bit-for-bit',
        'reference':'stored retrain signals of the source run; power from its original-vs-retrain rows'})
    sig={n:{} for n in requests}; jobs=[]
    with I.job_runner(SOURCE,workers) as run_jobs:
        for name,s,margins,seconds in run_jobs(_reset,[(n,s) for n in requests for s in seeds]):
            sig[name][s]=margins; jobs.append({'request':name,'seed':s,'seconds':seconds})
    print(f'models built ({time.perf_counter()-started:.0f}s)',flush=True)
    dims={}; checks={}
    for name,req in requests.items():
        stored,sizes,stored_seeds=I.load_signals(SOURCE,name); assert stored_seeds==seeds,name
        np.testing.assert_array_equal(sizes,feature_units(req['forget'][0])[1])
        dims[name]={'local':dead_dims(req['target_retain'][0]).tolist(),'global':dead_dims(*[b[0] for b in req['retained_blocks']]).tolist()}
        for arm in CONFIG['resets']:
            if not dims[name][arm.split('@')[1]]:
                same=all(np.array_equal(sig[name][s][arm],stored['retain_only'][i]) for i,s in enumerate(seeds))
                checks[f'{name}/{arm}']=same
                if not same: raise AssertionError(f'{name}/{arm}: empty reset differs from the stored retain-only')
        (output/name).mkdir()
        np.savez_compressed(output/name/'signals.npz',seeds=np.array(seeds),models=np.array(CONFIG['resets']),
                            signals=np.stack([[sig[name][s][k] for k in CONFIG['resets']] for s in seeds]))
    table,scores=analyse(output,requests,seeds,CONFIG['permutations'])
    table.to_csv(output/'distinguisher.csv',index=False); scores.to_csv(output/'scores.csv',index=False)
    pd.DataFrame(jobs).to_csv(output/'jobs.csv',index=False); write_json(output/'dims.json',dims)
    write_json(output/'status.json',{'completed':True,'seeds':len(seeds),'requests':list(requests),'empty_dims_checks':checks,
        'workers':workers,'total_seconds':time.perf_counter()-started})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--workers',type=int,default=1,help='processes for rebuilding models; results do not depend on it')
    a=p.parse_args(); run(a.output,a.workers)
