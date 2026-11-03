"""Model-level distinguishing test (U-LiRA-style): can an attacker who knows F tell an unlearned model from a retrain?

For the 5 deployable-forget requests and 40 seeds, the original, retrain, retain-only and three joint_aux_prob arms
(standard recovery) are observed through their logit margins on the distinct feature vectors of F. A Gaussian
likelihood-ratio distinguisher with per-vector means and pooled variances is fitted on all seeds but one and scored on
the held-out seed. The original is the positive control; two disjoint retrain groups are the negative control.
Seeds 10042-10051 are rebuilt from the stored clouds and checked against the deployable run; the new seeds use the same
training procedures, checked first by re-training stored seed-10042 models bit-for-bit.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from contextlib import contextmanager
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.experiment_context import Context,finalize,use_run_device
from scripts import run_canary_mia as M, run_deployable_forget as D, run_forget_signal as R
from src.data.prepare import digest,write_json
from src.eval.indistinguishability import distinguish
from src.eval.lira import feature_units
from src.eval.membership_validation import logit_margin,stable_baselines
from src.fl.fedavg_core import state,fedavg
from src.fu.neighbor_reference import knn_indices

CONFIG={'seeds':40,'first_new_seed':10052,'arms':['ga_adam@joint_aux_prob','ga_proj@joint_aux_prob','nkd_proj@joint_aux_prob'],
        'variants':['per_unit_lr','mean_margin'],'primary':'per_unit_lr','permutations':1000,'bootstrap':2000,'rng_seed':0,'alpha':.05,
        'observation':'logit margin z_attack - z_normal on each distinct feature vector of F'}
MODELS=['original','retrain','retain_only',*CONFIG['arms']]
COMPARATORS=['original','retain_only',*CONFIG['arms'],'retrain_split']
CHECK_SEED=10042
CHECKS=[('clean',None),('canary','canary_c3_500'),('canary','canary_c0_500'),('retrain','client4_random_stratified_0.5')]

def base_kinds(requests):
    """Models trained from scratch for every new seed: the clean original, canary originals and natural retrains."""
    return [('clean',None)]+[('canary' if r['kind']=='canary' else 'retrain',n) for n,r in requests.items()]

def pair_kinds(name,req):
    """(original, retrain) kinds of a request: canary retrains are the clean originals on D."""
    return (('canary',name),('clean',None)) if req['kind']=='canary' else (('clean',None),('retrain',name))

def blocks_for(requests,kind,name):
    if kind=='clean': return next(r for r in requests.values() if r['kind']=='canary')['retained_blocks']
    return requests[name]['original_blocks' if kind=='canary' else 'retained_blocks']

def stored(kind,name,s):
    """Root and checkpoint of a stored model of an existing seed; the root's artifact hashes cover it."""
    if kind=='clean': return R.SEED_FLOOR,Path('checkpoints')/f'original_seed{s}.pt'
    if kind=='retrain': return R.SEED_FLOOR,Path('checkpoints')/f'{name}_retrain_seed{s}.pt'
    rel=Path('checkpoints')/f'{name}_original_seed{s}.pt'
    return (R.IDENTIFIABLE if (R.IDENTIFIABLE/rel).exists() else D.FORGET_SIGNAL),rel

def new_path(root,kind,name,s): return root/'checkpoints'/(f'{kind}_seed{s}.pt' if name is None else f'{kind}_{name}_seed{s}.pt')

_HASHES={}
def load_stored(ctx,kind,name,s):
    root,rel=stored(kind,name,s)
    if root not in _HASHES: _HASHES[root]=json.loads((root/'artifact_hashes.json').read_text())
    return R.load(ctx,R.checked(root,rel,_HASHES[root]))

def forget_ce(m,req):
    """Forget CE exactly as the deployable run's Scorer records it."""
    return float(-stable_baselines(req['forget'][1],logit_margin(m,req['forget'][0]))['negative_loss'].mean())

_W={}
def _init(root):
    ctx=Context(); _W.update(ctx=ctx,requests=R.build_requests(ctx),root=root,cache={})

def model_of(kind,name,s):
    return load_stored(_W['ctx'],kind,name,s) if s<CONFIG['first_new_seed'] else R.load(_W['ctx'],new_path(_W['root'],kind,name,s))

def request_state(name):
    if name not in _W['cache']:
        ctx,req=_W['ctx'],_W['requests'][name]; fw,rw,_,recovery=R.workers_for(req); pools=D.pools_for(ctx,req); fx=req['forget'][0]
        unit,_=feature_units(fx)
        _W['cache'][name]={'fw':fw,'rw':rw,'recovery':recovery,'pools':pools,'units':fx[np.unique(unit,return_index=True)[1]],
                           'neighbors':{k:knn_indices(fx,p,D.CONFIG['knn_k']) for k,p in pools.items()}}
    return _W['cache'][name]

def _train(job):
    tag,kind,name,s=job; t0=time.perf_counter()
    m=R.train(_W['ctx'],blocks_for(_W['requests'],kind,name),s)
    return tag,kind,name,s,state(m),time.perf_counter()-t0

def _unlearn(job):
    """All models of one request and seed, observed on the distinct vectors of F, plus their forget CE for the rebuild check."""
    name,s=job; t0=time.perf_counter(); ctx,req=_W['ctx'],_W['requests'][name]; st=request_state(name)
    fl,rounds=ctx.cfg['fl'],ctx.cfg['recovery_rounds']; (ok,on),(rk,rn)=pair_kinds(name,req); orig=model_of(ok,on,s)
    ref,_,_=D.reference_model(ctx,req,s)
    _,levels,targets=D.references(orig,req['forget'][0],req['forget'][1],st['pools'],st['neighbors'],None,None,None,st['fw'].device,ref)
    models={'original':orig,'retrain':model_of(rk,rn,s),'retain_only':fedavg(orig,st['recovery'],fl,rounds,s+900000)[0]}
    for arm in CONFIG['arms']:
        op,stop=arm.split('@'); models[arm]=D.standard_arm(orig,st['fw'],st['rw'],st['recovery'],fl,rounds,s,op,stop,levels,targets)
    return name,s,{k:logit_margin(m,st['units']) for k,m in models.items()},{k:forget_ce(m,req) for k,m in models.items()},time.perf_counter()-t0

@contextmanager
def job_runner(root,workers):
    """Jobs depend only on their seeds and stored inputs, so worker processes give the same models as one process."""
    if workers==1:
        _init(root); yield map
    else:
        with ProcessPoolExecutor(workers,initializer=_init,initargs=(root,)) as ex: yield ex.map

def stored_ce(deploy,name,s,model):
    if model in ('original','retrain'): d=deploy['clouds.csv']; q=d[(d.request==name)&(d.seed==s)&(d.cloud==model)]
    elif model=='retain_only': d=deploy['controls.csv']; q=d[(d.request==name)&(d.seed==s)&(d.control_steps==0)&(d.stage=='post')]
    else: d=deploy['fu_runs.csv']; q=d[(d.request==name)&(d.seed==s)&(d.arm==model)&(d.stage=='post')]
    return float(q.forget_ce.iloc[0])

def observations(sig,sizes,variant):
    """[n_seeds, U] margins, or the row-weighted mean margin over F as one observation."""
    return sig if variant=='per_unit_lr' else (sig@sizes/sizes.sum())[:,None]

def analyse(name,ri,signals,sizes,seeds,permutations):
    """Every comparator against retrain, plus the retrain-vs-retrain negative control on disjoint seed halves."""
    rows,scores=[],[]; half=len(seeds)//2
    for ci,comp in enumerate(COMPARATORS):
        if comp=='retrain_split': c,r,sd=signals['retrain'][half:2*half],signals['retrain'][:half],seeds[half:2*half]
        else: c,r,sd=signals[comp],signals['retrain'],seeds
        for vi,variant in enumerate(CONFIG['variants']):
            rng=np.random.default_rng([CONFIG['rng_seed'],ri,ci,vi])
            res,sc,sr=distinguish(observations(c,sizes,variant),observations(r,sizes,variant),permutations,CONFIG['bootstrap'],rng)
            rows.append({'request':name,'comparator':comp,'variant':variant,'primary':variant==CONFIG['primary'],'seeds':len(c),**res})
            scores.extend({'request':name,'comparator':comp,'variant':variant,'seed':int(s),'score_comparator':a,'score_retrain':b}
                          for s,a,b in zip(sd,sc,sr))
    return rows,scores

def verdicts(table):
    """Power per request and variant: the original is distinguished from retrain at alpha."""
    orig=table[table.comparator=='original'].set_index(['request','variant']).p_value<CONFIG['alpha']
    table['power']=[bool(orig[(q,v)]) for q,v in zip(table.request,table.variant)]
    table['verdict']=np.where(~table.power,'INCONCLUSIVE',np.where(table.p_value<CONFIG['alpha'],'DISTINGUISHABLE','NOT_DISTINGUISHED'))
    return table,{q:bool(orig[(q,CONFIG['primary'])]) for q in table.request.unique()}

def load_signals(root,name):
    with np.load(root/name/'signals.npz') as z:
        return {k:z['signals'][:,i] for i,k in enumerate(z['models'].tolist())},z['sizes'],z['seeds'].tolist()

def run(output,n_seeds,permutations,workers):
    use_run_device(M.DEPLOYABLE); M.check_deployable(); D.check_source(); started=time.perf_counter()
    existing=json.loads((M.DEPLOYABLE/'manifest.json').read_text())['protocol']['seeds']
    if existing!=list(range(CHECK_SEED,CONFIG['first_new_seed'])): raise ValueError('Unexpected deployable-forget seeds')
    seeds=(existing+list(range(CONFIG['first_new_seed'],CONFIG['first_new_seed']+n_seeds-len(existing))))[:n_seeds]
    new=[s for s in seeds if s>=CONFIG['first_new_seed']]
    ctx=Context(); requests=R.build_requests(ctx)
    ctx.manifest_for(output,'indistinguishability',{**CONFIG,'seeds':n_seeds,'permutations':permutations,'seed_list':seeds,
        'models':MODELS,'comparators':COMPARATORS,'deployable_source':str(M.DEPLOYABLE),
        'deployable_artifact_hashes_sha256':digest(M.DEPLOYABLE/'artifact_hashes.json'),
        'procedure_check':f'new seeds are trained only after seed {CHECK_SEED} re-trains bit-for-bit to the stored {CHECKS}',
        'rebuild_check':'existing seeds: every model matches the deployable run forget CE (clouds, controls, fu_runs) within 1e-9',
        'distinguisher':'leave-one-seed-out Gaussian LR, per-observation means, pooled variance per group (src/eval/indistinguishability.py)',
        'negative_control':'retrains of the second half of the seeds vs the first half, paired by index',
        'p_value':'one-sided, within-seed label swaps with the leave-one-out fit redone',
        'verdicts':'power iff original vs retrain p < alpha; DISTINGUISHABLE iff p < alpha; no multiplicity correction (conservative)'})
    (output/'checkpoints').mkdir()
    deploy={f:pd.read_csv(M.DEPLOYABLE/f) for f in ['clouds.csv','controls.csv','fu_runs.csv']}
    checks,train_log,job_log,rebuild={},[],[],[]; sig={n:{} for n in requests}
    with job_runner(output,workers) as run_jobs:
        jobs=[('check',k,n,CHECK_SEED) for k,n in CHECKS]+[('new',k,n,s) for s in new for k,n in base_kinds(requests)]
        for tag,kind,name,s,st,seconds in run_jobs(_train,jobs):
            if tag=='check':
                ref=load_stored(ctx,kind,name,s).state_dict()
                checks[f'{kind}/{name}']=all(torch.equal(ref[k].cpu(),v) for k,v in st.items())
                if not checks[f'{kind}/{name}']: raise AssertionError(f'Training does not reproduce the stored {kind}/{name} seed {s}')
            else:
                torch.save({'state_dict':st,'seed':s},new_path(output,kind,name,s))
                train_log.append({'kind':kind,'name':name,'seed':s,'seconds':seconds})
        print(f'procedure checks passed; {len(train_log)} new models ({time.perf_counter()-started:.0f}s)',flush=True)
        for name,s,margins,ces,seconds in run_jobs(_unlearn,[(n,s) for n in requests for s in seeds]):
            sig[name][s]=margins; job_log.append({'request':name,'seed':s,'seconds':seconds})
            if s in existing:
                for k,v in ces.items():
                    diff=abs(v-stored_ce(deploy,name,s,k)); rebuild.append({'request':name,'seed':s,'model':k,'abs_diff':diff})
                    if diff>1e-9: raise AssertionError(f'{name}/{s}/{k}: rebuilt model differs from the deployable run')
        print(f'models built ({time.perf_counter()-started:.0f}s)',flush=True)
    rows,scores=[],[]
    for ri,(name,req) in enumerate(requests.items()):
        _,sizes=feature_units(req['forget'][0]); (output/name).mkdir()
        np.savez_compressed(output/name/'signals.npz',seeds=np.array(seeds),models=np.array(MODELS),sizes=sizes,
                            signals=np.stack([[sig[name][s][k] for k in MODELS] for s in seeds]))
        signals,sizes,_=load_signals(output,name)
        r,s=analyse(name,ri,signals,sizes,seeds,permutations); rows+=r; scores+=s
        print(f'{name}: analysed ({time.perf_counter()-started:.0f}s)',flush=True)
    table,power=verdicts(pd.DataFrame(rows))
    table.to_csv(output/'distinguisher.csv',index=False); pd.DataFrame(scores).to_csv(output/'scores.csv',index=False)
    pd.DataFrame(rebuild).to_csv(output/'rebuild_checks.csv',index=False); pd.DataFrame(train_log).to_csv(output/'new_models.csv',index=False)
    pd.DataFrame(job_log).to_csv(output/'jobs.csv',index=False); write_json(output/'procedure_checks.json',checks)
    write_json(output/'status.json',{'completed':True,'seeds':len(seeds),'new_seeds':len(new),'requests':list(requests),
        'procedure_checks_passed':all(checks.values()),'rebuilt_models_match_deployable_run':True,'power':power,'workers':workers,
        'total_seconds':time.perf_counter()-started,'unlearning_success_claimed':False})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=CONFIG['seeds']); p.add_argument('--permutations',type=int,default=CONFIG['permutations'])
    p.add_argument('--workers',type=int,default=1,help='processes for model training and unlearning; results do not depend on it')
    a=p.parse_args(); run(a.output,a.seeds,a.permutations,a.workers)
