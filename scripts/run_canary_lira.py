"""Per-example LiRA (Carlini et al. 2022) on the canary requests, scored on the deployable-forget models.

The pool of each canary request is its 500 canaries plus the 500 matched nonmembers of the canary MIA. Rows
with identical features form one membership unit; units holding both canaries and nonmembers have no defined
membership and are left out of the target cohorts. Shadows come in complementary pairs over the units, each
with 500 +- 25 pool rows appended to the target client as the canaries are in the original, so every unit is IN
for exactly half of them. Leave-one-shadow-out is the control with known membership; on the target side the
original must score above retrain, paired over seeds.
"""
import argparse
import json
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.experiment_context import Context,finalize,use_run_device
from scripts import run_canary_mia as M, run_deployable_forget as D, run_forget_signal as R
from src.data.prepare import digest,write_json
from src.eval.fu_decision import mean_ci
from src.eval.lira import LiRA,lira_signal,evaluate_mia,feature_units,complementary_masks,loo_scores
from src.fl.fedavg_core import model,state,fedavg,ClientWorker

CONFIG={'shadows':64,'pool_rows':500,'pool_row_tolerance':25,'mask_seed_offset':5100,'shadow_seed_offset':2000000,
        'variants':{'online_global':['online','global'],'online_per_example':['online','per_example'],
                    'offline_global':['offline','global'],'loss':None},
        'primary':'online_global','control_fprs':[.001,.01,.05],'target_fprs':[.01,.05],
        'signal':'z_normal - z_attack in float64 (lira_signal with the canary label, normal)'}
MODELS=['original','retrain','retain_only',*M.CONFIG['arms']]

def signal(m,x): return lira_signal(m,x,np.zeros(len(x),dtype=np.int64))

def unit_roles(unit,member,n_units):
    """'member' / 'nonmember' for units on one side only, 'shared' for units holding both."""
    has_m=np.bincount(unit[member],minlength=n_units)>0; has_n=np.bincount(unit[~member],minlength=n_units)>0
    return np.where(has_m&has_n,'shared',np.where(has_m,'member','nonmember'))

def pool_for(ctx,req):
    """Pool rows (canaries, then matched nonmembers), their units, unit roles and one feature vector per unit."""
    members,nonmembers,picked=M.canary_sets(ctx,req); x=np.r_[members,nonmembers]
    unit,sizes=feature_units(x); member=np.arange(len(x))<len(members)
    first=np.unique(unit,return_index=True)[1]
    return {'x':x,'unit':unit,'sizes':sizes,'role':unit_roles(unit,member,len(sizes)),'unit_x':x[first],'picked':picked}

def masks_for(ctx,req,pool,shadows):
    return complementary_masks(pool['sizes'],shadows,CONFIG['pool_rows'],CONFIG['pool_row_tolerance'],
                               ctx.cfg['seed']+CONFIG['mask_seed_offset']+req['target'])

def shadow_jobs(ctx,req,pool,masks):
    return [(k,ctx.cfg['seed']+CONFIG['shadow_seed_offset']+1000*req['target']+k,np.flatnonzero(masks[k][pool['unit']]))
            for k in range(len(masks))]

def train_shadow(blocks,target,x,seed,hidden,fl):
    """FedAvg on the base blocks with pool rows x (label normal) appended to the target client, as canaries are in the original."""
    ids=np.array([f'pool_{i}' for i in range(len(x))])
    workers=[ClientWorker(j,*((np.r_[b[0],x],np.r_[b[1],np.zeros(len(x),dtype=np.int64)],np.r_[b[2],ids]) if j==target else b))
             for j,b in enumerate(blocks)]
    return fedavg(model(blocks[0][0].shape[1],hidden,seed),workers,fl,fl['rounds'],seed)

_W={}
def _init(blocks,pool_x,unit_x,target,hidden,fl,threads):
    _W.update(blocks=blocks,pool_x=pool_x,unit_x=unit_x,target=target,hidden=hidden,fl=fl)
    torch.set_num_threads(threads); torch.use_deterministic_algorithms(True)

def _shadow(job):
    k,seed,rows=job; started=time.perf_counter()
    m,logs=train_shadow(_W['blocks'],_W['target'],_W['pool_x'][rows],seed,_W['hidden'],_W['fl'])
    return k,signal(m,_W['unit_x']),state(m),time.perf_counter()-started,sum(c['steps'] for r in logs for c in r['clients'])

def run_shadows(ctx,req,pool,jobs,workers):
    """Each shadow depends only on its seed and rows, so worker processes give the same models as one process."""
    init=(req['retained_blocks'],pool['x'],pool['unit_x'],req['target'],ctx.cfg['hidden'],ctx.cfg['fl'],ctx.cfg['threads'])
    if workers==1:
        _init(*init); yield from map(_shadow,jobs); return
    with ProcessPoolExecutor(workers,initializer=_init,initargs=init) as ex: yield from ex.map(_shadow,jobs)

def attacks(signals,masks):
    return {v:None if spec is None else LiRA(*spec).fit(signals,masks) for v,spec in CONFIG['variants'].items()}

def control_rows(name,signals,masks,role):
    cols=np.flatnonzero(role!='shared'); rows=[]
    for v,spec in CONFIG['variants'].items():
        pairs=loo_scores(signals,masks,spec,cols); a,lo,hi=mean_ci([evaluate_mia(m,n,())['auc'] for m,n in pairs])
        pooled=evaluate_mia(np.concatenate([m for m,_ in pairs]),np.concatenate([n for _,n in pairs]),CONFIG['control_fprs'])
        rows.append({'request':name,'variant':v,'shadows':len(signals),'units':len(cols),'auc_mean':a,'auc_ci_low':lo,'auc_ci_high':hi,
                     'control_power':bool(lo>.5),**{k:x for k,x in pooled.items() if k.startswith('roc_')}})
    return rows

def target_rows(name,signals,masks,role,targets):
    """targets: {(seed, model): signal on the units}."""
    fitted=attacks(signals,masks); m,n=role=='member',role=='nonmember'; rows=[]
    for (s,k),phi in targets.items():
        for v,a in fitted.items():
            sc=phi if a is None else a.score(phi)
            rows.append({'request':name,'seed':s,'model':k,'variant':v,**evaluate_mia(sc[m],sc[n],CONFIG['target_fprs'])})
    return rows

def summarize(rows,control):
    tprs=[c for c in rows.columns if c.startswith('roc_tpr')]; out=[]
    for (name,v),g in rows.groupby(['request','variant'],sort=False):
        auc=g.pivot(index='seed',columns='model',values='auc'); base=auc['original']-auc['retrain']
        c=control[(control.request==name)&(control.variant==v)].iloc[0]
        power=bool(c.control_power and mean_ci(base)[1]>0)
        for k in [k for k in MODELS if k in auc.columns]:
            a,r,o=mean_ci(auc[k]),mean_ci(auc[k]-auc['retrain']),mean_ci(auc['original']-auc[k])
            out.append({'request':name,'variant':v,'primary':v==CONFIG['primary'],'model':k,'seeds':len(auc),
                'auc_mean':a[0],'auc_ci_low':a[1],'auc_ci_high':a[2],
                'auc_minus_retrain_mean':r[0],'auc_minus_retrain_ci_low':r[1],'auc_minus_retrain_ci_high':r[2],
                'original_minus_auc_mean':o[0],'original_minus_auc_ci_low':o[1],'original_minus_auc_ci_high':o[2],
                'leak_remaining':r[0]/base.mean() if base.mean() else float('nan'),
                **{f'{t}_mean':g[g.model==k][t].mean() for t in tprs},
                'control_power':bool(c.control_power),'attack_has_power':power,
                'verdict_vs_retrain':'MIA_INCONCLUSIVE' if not power else 'RETRAIN_LIKE' if r[1]<=0<=r[2] else 'DIFFERS_FROM_RETRAIN',
                'verdict_vs_original':'MIA_INCONCLUSIVE' if not power else 'LOWER_THAN_ORIGINAL' if o[1]>0 else 'NOT_LOWER'})
    summary=pd.DataFrame(out); primary=summary[summary.primary]
    return summary,{q:bool(s.attack_has_power.iloc[0]) for q,s in primary.groupby('request',sort=False)}

def pool_audit(pool,masks,req):
    rows=np.array([pool['sizes'][m].sum() for m in masks]); role=pool['role']; member=np.arange(len(pool['x']))<len(req['forget'][0])
    shared=role[pool['unit']]=='shared'
    return {'target':req['target'],'pool_rows':len(pool['x']),'units':len(pool['sizes']),'max_unit_rows':int(pool['sizes'].max()),
            **{f'{r}_units':int((role==r).sum()) for r in ['member','nonmember','shared']},
            'canary_rows_in_shared_units':int((shared&member).sum()),'nonmember_rows_in_shared_units':int((shared&~member).sum()),
            'shadow_pool_rows_min':int(rows.min()),'shadow_pool_rows_max':int(rows.max())}

def run(output,shadows,seeds_limit,workers):
    use_run_device(M.DEPLOYABLE); M.check_deployable(); D.check_source(); started=time.perf_counter()
    ctx=Context(); seeds=json.loads((M.DEPLOYABLE/'manifest.json').read_text())['protocol']['seeds'][:seeds_limit]
    ctx.manifest_for(output,'canary_lira',{**CONFIG,'shadows':shadows,'seeds':seeds,'models':MODELS,
        'deployable_source':str(M.DEPLOYABLE),'deployable_artifact_hashes_sha256':digest(M.DEPLOYABLE/'artifact_hashes.json'),
        'pool':'500 canaries + the 500 matched nonmembers of run_canary_mia (nonmember_count, seed offset as there)',
        'unit':'distinct feature vector; units holding both canaries and nonmembers are excluded from the target cohorts',
        'masks':'complementary pairs over units by fair coins, redrawn until both halves hold pool_rows +- tolerance rows',
        'shadow_training':'FedAvg as the original on D with the IN rows appended to the target client; seed + offset + 1000*target + k',
        'control':'leave-one-shadow-out over all shadows on the target-cohort units; power iff mean-AUC lower 95% bound > 0.5',
        'power_rule':'control power and the paired 95% CI over seeds of AUC(original) - AUC(retrain) above 0',
        'verdicts':'vs retrain: paired CI of AUC(model) - AUC(retrain) contains 0 -> RETRAIN_LIKE; vs original: '
                   'paired CI of AUC(original) - AUC(model) above 0 -> LOWER_THAN_ORIGINAL'})
    stored=pd.read_csv(M.DEPLOYABLE/'fu_runs.csv'); controls=pd.read_csv(M.DEPLOYABLE/'controls.csv')
    requests={k:v for k,v in R.build_requests(ctx).items() if v['kind']=='canary'}
    control=[]; rows=[]; shadow_log=[]; audit={}
    for name,req in requests.items():
        out=output/name; (out/'shadows').mkdir(parents=True)
        pool=pool_for(ctx,req); masks=masks_for(ctx,req,pool,shadows); jobs=shadow_jobs(ctx,req,pool,masks)
        signals=np.empty((shadows,len(pool['sizes'])))
        for k,phi,st,seconds,steps in run_shadows(ctx,req,pool,jobs,workers):
            signals[k]=phi; torch.save({'state_dict':st,'seed':jobs[k][1]},out/'shadows'/f'shadow{k}.pt')
            shadow_log.append({'request':name,'shadow':k,'seed':jobs[k][1],'pool_rows_in':len(jobs[k][2]),
                               'units_in':int(masks[k].sum()),'optimizer_steps':steps,'wall_seconds':seconds})
        print(f'{name}: {shadows} shadows ({time.perf_counter()-started:.0f}s)',flush=True)
        targets={}
        for s,models in M.rebuild(ctx,name,req,seeds,stored,controls):
            targets.update({(s,k):signal(m,pool['unit_x']) for k,m in models.items()})
        np.savez_compressed(out/'pool.npz',unit=pool['unit'],sizes=pool['sizes'],role=pool['role'],masks=masks,
                            nonmember_row_ids=ctx.tr['row_id'][pool['picked']],source_row_ids=np.array(req['audit']['source_row_ids']))
        np.savez_compressed(out/'signals.npz',shadows=signals,target_seeds=np.array([s for s,_ in targets]),
                            target_models=np.array([k for _,k in targets]),targets=np.array(list(targets.values())))
        control.extend(control_rows(name,signals,masks,pool['role'])); rows.extend(target_rows(name,signals,masks,pool['role'],targets))
        audit[name]=pool_audit(pool,masks,req)
        print(f'{name}: scored ({time.perf_counter()-started:.0f}s)',flush=True)
    control=pd.DataFrame(control); rows=pd.DataFrame(rows); summary,power=summarize(rows,control)
    control.to_csv(output/'control.csv',index=False); rows.to_csv(output/'target_rows.csv',index=False)
    summary.to_csv(output/'summary.csv',index=False); pd.DataFrame(shadow_log).to_csv(output/'shadows.csv',index=False)
    write_json(output/'pool_audit.json',audit)
    write_json(output/'status.json',{'completed':True,'shadows':shadows,'seeds':len(seeds),'requests':list(requests),
        'attack_has_power':power,'rebuilt_models_match_stored_run':True,'workers':workers,
        'total_seconds':time.perf_counter()-started,'unlearning_success_claimed':False})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shadows',type=int,default=CONFIG['shadows']); p.add_argument('--seeds',type=int,default=None,help='first N seeds (smoke runs)')
    p.add_argument('--workers',type=int,default=1,help='shadow-training processes; results do not depend on it')
    a=p.parse_args(); run(a.output,a.shadows,a.seeds,a.workers)
