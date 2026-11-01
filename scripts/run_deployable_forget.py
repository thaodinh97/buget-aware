"""Deployable forget stop (neighbour reference) and a forget signal that survives recovery (retain-orthogonal
projection, interleaved forget-recovery), judged by the forget-signal decision rule on the same requests and clouds."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.special import expit
from scripts.experiment_context import Context,finalize
from scripts.run_forget_signal import (CONTROL,SEED_FLOOR,build_requests,cloud_models,Scorer,workers_for,decide,gate_entry)
from src.data.prepare import digest,write_json
from src.eval.membership_validation import logit_margin
from src.fl.fedavg_core import model,fedavg,ClientWorker,get_device
from src.fu.bounded_forget import local_forget,retain_matched_many,matched_steps,forget_ce,forget_prob
from src.fu.neighbor_reference import knn_indices,neighbor_margin,reference_level,reference_prob,soft_targets
from src.fu.recovery_schedule import interleaved

FORGET_SIGNAL=Path('runs/forget_signal_seed42')
PROJ={'threshold':.999999,'max_rows':4000}
CONFIG={'knn_k':10,'pools':{'knn_local':['target_retain'],'knn_aux':['target_retain','validation']},
        'local':{**CONTROL,'max_steps':300,'check_every':1,'ce_margin':20.},
        'ops':{'ga_adam':{'objective':'ga','alpha':0.},'ga_proj':{'objective':'ga','alpha':0.,'projection':PROJ},
               'nkd_proj':{'objective':'soft_target','alpha':0.,'projection':PROJ}},
        'reference':{'pool':['target_retain','validation'],'lr':1e-3,'local_epochs':5,'batch_size':512,'seed_offset':700000},
        'stops':{'ga_adam':['knn_local','knn_aux','knn_aux_prob','ref_aux','ref_aux_prob','joint_aux_prob','retrain_oracle'],
                 'ga_proj':['knn_local','knn_aux','knn_aux_prob','ref_aux','ref_aux_prob','joint_aux_prob','retrain_oracle'],
                 'nkd_proj':['knn_local','knn_aux','knn_aux_prob','ref_aux','ref_aux_prob','joint_aux_prob']},
        'recoveries':['standard','interleaved'],'control':CONTROL,'control_round_above':1000,'control_round':64,'min_targets':2}

def control_budget(backward_samples):
    """Matched retain-only steps; budgets above 1000 steps round up to a multiple of 64, so the control never gets less."""
    k=matched_steps(backward_samples,CONTROL['batch_size']); r=CONFIG['control_round']
    return k if k<=CONFIG['control_round_above'] else -(-k//r)*r

def charged(stop): return stop.startswith(('ref_aux','joint_aux'))

def check_source():
    for name,sha in json.loads((FORGET_SIGNAL/'artifact_hashes.json').read_text()).items():
        if digest(FORGET_SIGNAL/name)!=sha: raise ValueError(f'Changed forget-signal artifact {name}')

def parts_for(ctx,req):
    va=ctx.data['validation']; return {'target_retain':req['target_retain'][:2],'validation':(va['X'],va['y'])}

def pools_for(ctx,req):
    parts=parts_for(ctx,req); return {k:np.concatenate([parts[p][0] for p in v]) for k,v in CONFIG['pools'].items()}

def reference_model(ctx,req,s):
    """F-naive reference: a fresh model trained locally on R_target plus the auxiliary set, never on F.
    Returns the model, its forward+backward rows (charged to the arms that use it) and seconds."""
    c=CONFIG['reference']; parts=parts_for(ctx,req); x=np.concatenate([parts[p][0] for p in c['pool']])
    y=np.concatenate([parts[p][1] for p in c['pool']]); seed=s+c['seed_offset']; t0=time.perf_counter()
    m=model(x.shape[1],ctx.cfg['hidden'],seed)
    weights,_,_=ClientWorker(req['target'],x,y,np.arange(len(y))).train(m,{k:c[k] for k in ['lr','local_epochs','batch_size']},seed)
    m.load_state_dict(weights); return m,c['local_epochs']*len(y),time.perf_counter()-t0

def references(orig,fx,fy,pools,neighbors,retrain_ce,retrain_p,i,device,ref=None):
    """Per-seed stop levels (CE, and true-label probability for *_prob) and soft targets.
    retrain_oracle is the retrain-cloud mean on F with seed i left out."""
    margins={k:neighbor_margin(orig,pools[k],neighbors[k]) for k in pools}
    if ref is not None:
        # joint: per row, whichever reference gives the true label the lower probability (both err towards under-forgetting).
        margins['ref_aux']=logit_margin(ref,fx); sign=2*np.asarray(fy)-1
        margins['joint_aux']=sign*np.minimum(sign*margins['knn_aux'],sign*margins['ref_aux'])
    levels={**{k:reference_level(fy,v) for k,v in margins.items()},**{f'{k}_prob':reference_prob(fy,v) for k,v in margins.items()},
            'retrain_oracle':float(np.delete(retrain_ce,i).mean()),'retrain_oracle_prob':float(np.delete(retrain_p,i).mean())}
    return margins,levels,{k:soft_targets(v,device) for k,v in margins.items()}

def arm_runs(orig,fw,rw,recovery,fl,rounds,s,op,stop,levels,targets,overhead=(0,0.)):
    """Standard and interleaved variants of one local arm; the interleaved first stage reuses the standard one.
    overhead = (rows, seconds) of building the stop reference, charged once to both variants."""
    cfg={**CONFIG['local'],**CONFIG['ops'][op],'stop_statistic':'prob' if stop.endswith('_prob') else 'ce'}
    tg=targets[stop.removesuffix('_prob')] if cfg['objective']=='soft_target' else None
    run_stage=lambda start:local_forget(start,fw,rw,cfg,s,levels[stop],tg)
    t0=time.perf_counter(); pre,_,info=run_stage(orig); local=time.perf_counter()-t0+overhead[1]
    info={**info,'backward_samples':info['backward_samples']+overhead[0],'reference_backward_samples':overhead[0]}
    t0=time.perf_counter(); post,_=fedavg(pre,recovery,fl,rounds,s+900000)
    out=[('standard',pre,post,info,local,time.perf_counter()-t0)]
    later=[]
    def stage(m):
        if not later: later.append(0.); return pre,info
        t1=time.perf_counter(); m,_,inf=run_stage(m); later.append(time.perf_counter()-t1); return m,inf
    t0=time.perf_counter(); post,infos=interleaved(orig,stage,recovery,fl,rounds,s+900000); total=time.perf_counter()-t0
    comb={**info,'steps':sum(x['steps'] for x in infos),'backward_samples':sum(x['backward_samples'] for x in infos),
          'check_samples':sum(x['check_samples'] for x in infos),'stop_end_ce':infos[-1]['stop_end_ce'],
          'later_stage_steps':'/'.join(str(x['steps']) for x in infos[1:]),
          'later_stop_reasons':'/'.join(x['stop_reason'] for x in infos[1:])}
    out.append(('interleaved',pre,post,comb,local+sum(later),total-sum(later)))
    return out

def run(output,seeds_limit=None):
    ctx=Context(); started=time.perf_counter(); check_source()
    seeds=json.loads((SEED_FLOOR/'manifest.json').read_text())['protocol']['seeds'][:seeds_limit]
    print(f'Using device: {get_device()}',flush=True)
    ctx.manifest_for(output,'deployable_forget',{**CONFIG,'seeds':seeds,'forget_signal_source':str(FORGET_SIGNAL),
        'forget_signal_artifact_hashes_sha256':digest(FORGET_SIGNAL/'artifact_hashes.json'),
        'stop_rule':'CE(F) >= level; knn_* level = mean CE of the original mean logit margin over the k nearest pool rows of each f; '
                    'ref_aux level = CE(F) of a fresh model trained on R_target + validation (its training rows are charged to the arm); '
                    '*_prob: same reference, but the statistic is the mean true-label probability on F (stop once <= level); '
                    'retrain_oracle = retrain-cloud mean CE(F), same seed left out (oracle, not deployable)',
        'aux_rule':'validation split stands in for an approved auxiliary labelled set; test is never used',
        'projection_rule':'per-layer SVD basis of augmented inputs on R_target at the stage start; gradients and Adam updates projected off it',
        'interleaved_rule':'stage -> 1 round -> stage -> 1 round; same round seeds as standard recovery; F never inside FedAvg',
        'control_rule':'compute-matched Adam on CE(R_target), standard recovery, as forget_signal; budgets > 1000 steps rounded up to x64; '
                       'all budgets of a seed share one prefix-identical pass',
        'joint_rule':'joint_aux: per row the reference (knn_aux or ref_aux) with the lower true-label probability',
        'decision_rule':'src/eval/fu_decision.py via run_forget_signal.decide, unchanged','selection':'none; every arm reported'})
    requests=build_requests(ctx)
    write_json(output/'requests.json',{k:{'kind':v['kind'],'target':v['target'],'primary':v['primary'],'forget_count':len(v['forget'][1]),
        **{f'{p}_pool_size':len(x) for p,x in pools_for(ctx,v).items()}} for k,v in requests.items()})
    cloud_rows,gate,runs,controls,level_rows=[],{},[],[],[]
    for name,req in requests.items():
        clouds,source=cloud_models(ctx,name,req,seeds,FORGET_SIGNAL,False); score=Scorer(ctx,req,clouds['retrain'])
        values={k:[score(m,i) for i,m in enumerate(ms)] for k,ms in clouds.items()}
        cloud_rows.extend({'request':name,'cloud':k,'seed':s,**v} for k,vals in values.items() for s,v in zip(seeds,vals))
        gate[name]=gate_entry(values,req['target'],source)
        if not gate[name]['separable']: print(f'{name}: not admitted',flush=True); continue
        fw,rw,_,recovery=workers_for(req); fx,fy=req['forget'][0],req['forget'][1]
        pools=pools_for(ctx,req); t0=time.perf_counter()
        neighbors={k:knn_indices(fx,pool,CONFIG['knn_k']) for k,pool in pools.items()}
        knn_seconds=time.perf_counter()-t0
        retrain_ce,retrain_p=(np.array([f(m,fw) for m in clouds['retrain']]) for f in (forget_ce,forget_prob))
        print(f'{name}: neighbours in {knn_seconds:.1f}s',flush=True)
        for i,(s,orig) in enumerate(zip(seeds,clouds['original'])):
            ref,ref_rows,ref_seconds=reference_model(ctx,req,s)
            margins,levels,targets=references(orig,fx,fy,pools,neighbors,retrain_ce,retrain_p,i,fw.device,ref)
            others=np.delete(score.rF,i,0)
            level_rows.append({'request':name,'seed':s,'knn_seconds':knn_seconds,'ref_rows':ref_rows,'ref_seconds':ref_seconds,'original_forget_ce':forget_ce(orig,fw),'original_forget_prob':forget_prob(orig,fw),
                'original_mae_F':values['original'][i]['mae_F'],**{(k if k.endswith('_prob') else f'{k}_ce'):v for k,v in levels.items()},
                **{f'{k}_reference_mae_F':float(np.abs(expit(m)-others).mean()) for k,m in margins.items()}})
            budgets={0}
            for op,stops in CONFIG['stops'].items():
                for stop in stops:
                    overhead=(ref_rows,ref_seconds) if charged(stop) else (0,0.)
                    for kind,pre,post,info,local,rec in arm_runs(orig,fw,rw,recovery,ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s,op,stop,levels,targets,overhead):
                        steps=control_budget(info['backward_samples']); budgets.add(steps)
                        common={'request':name,'target':req['target'],'seed':s,'arm':f'{op}@{stop}'+('' if kind=='standard' else '+interleaved'),
                                'family':'local','op':op,'stop':stop,'recovery':kind,'control_steps':steps,
                                'local_seconds':local,'recovery_seconds':rec,**info}
                        for stage,m in [('pre',pre),('post',post)]: runs.append({**common,'stage':stage,**score(m,i)})
            t1=time.perf_counter(); matched=retain_matched_many(orig,rw,CONTROL,budgets,s); one_pass=time.perf_counter()-t1
            for steps,pre in sorted(matched.items()):
                post,_=fedavg(pre,recovery,ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s+900000)
                for stage,m in [('pre',pre),('post',post)]:
                    controls.append({'request':name,'seed':s,'control_steps':steps,'stage':stage,
                                     'local_seconds':one_pass*steps/max(budgets) if steps else 0.,**score(m,i)})
            print(f'{name}: seed {i+1}/{len(seeds)} ({time.perf_counter()-started:.0f}s)',flush=True)
    clouds=pd.DataFrame(cloud_rows); runs=pd.DataFrame(runs); controls=pd.DataFrame(controls)
    clouds.to_csv(output/'clouds.csv',index=False); runs.to_csv(output/'fu_runs.csv',index=False)
    controls.to_csv(output/'controls.csv',index=False); pd.DataFrame(level_rows).to_csv(output/'stop_levels.csv',index=False)
    write_json(output/'gate.json',gate)
    admitted={k:v['target'] for k,v in gate.items() if v['separable']}
    decision,verdict=decide(clouds,runs,controls,admitted,CONFIG['min_targets']) if len(runs) else (pd.DataFrame(),{})
    decision.to_csv(output/'decision.csv',index=False); write_json(output/'verdict.json',verdict)
    write_json(output/'status.json',{'completed':True,'seeds':len(seeds),'requests':list(requests),'requests_passing_gate':list(admitted),
        'targets_passing_gate':sorted(set(admitted.values())),'fu_rows':len(runs),'control_rows':len(controls),
        'arms_passing_post':sorted(k for k,v in verdict.items() if k.endswith('/post') and v['pass']),
        'total_seconds':time.perf_counter()-started,'selection_performed':False,'unlearning_success_claimed':False})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=None,help='use only the first N seed-floor seeds (smoke runs)')
    a=p.parse_args(); run(a.output,a.seeds)
