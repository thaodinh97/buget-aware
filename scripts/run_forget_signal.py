"""Forget-signal benchmark: bounded, target-stopped local forgetting, KD with a forget signal, and a
seed-noise decision rule against compute-matched retain-only controls over seeds and target clients."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scripts.experiment_context import Context,finalize
from src.data.canary import trigger_dims,stamp,make_canaries
from src.data.prepare import digest,write_json
from src.eval.fu_decision import paired_gain,utility_noninferior,local_step_check,overall
from src.eval.membership_validation import logit_margin,stable_baselines
from src.eval.metrics import utility
from src.eval.seed_floor import classify
from src.fl.fedavg_core import model,state,fedavg,ClientWorker,get_device
from src.fu.bounded_forget import local_forget,retain_matched,matched_steps,forget_ce,class_proportional
from src.fu.forget_kd import forget_kd

SEED_FLOOR=Path('runs/seed_floor_seed42'); IDENTIFIABLE=Path('runs/identifiable_fu_seed42')
CONTROL={'optimizer':'adam','lr':1e-3,'batch_size':128,'clip':5.}
CONFIG={'canary_clients':[3,0,4],'canary_count':500,'trigger_count':3,
        'natural_requests':['client4_random_stratified_0.5','client4_low_overlap_groups_0.5'],
        'nonmember_cap':2000,'control':CONTROL,
        'local':{**CONTROL,'max_steps':100,'check_every':1,'ce_margin':10.},
        'kd':{**CONTROL,'steps':100,'check_every':1,'ce_margin':10.,'temperature':2.,'hard_weight':.5,
              'forget_weight':.5,'scrub_ascent_steps':10,'scrub_gamma':1.},
        'objectives':{'ga_adam':{'objective':'ga','alpha':0.},'neggrad_plus':{'objective':'ga','alpha':.9},
                      'random_label':{'objective':'random_label','alpha':0.},'uniform':{'objective':'uniform','alpha':0.},
                      'ga_sgd':{'objective':'ga','alpha':0.,'optimizer':'sgd','lr':.02}},
        'stops':['nonmember','retrain_oracle'],
        'kd_arms':[['kd_retain','neggrad_plus@nonmember',None],['kd_teacher_on_f','neggrad_plus@nonmember','nonmember'],
                   ['bad_teacher',None,'nonmember'],['bad_teacher',None,'retrain_oracle'],
                   ['scrub',None,'nonmember'],['scrub',None,'retrain_oracle']],
        'min_targets':2}

def load(ctx,path):
    ck=torch.load(path,weights_only=True); m=model(ctx.tr['X'].shape[1],ctx.cfg['hidden'],ck['seed'])
    m.load_state_dict(ck['state_dict']); return m

def train(ctx,blocks,seed):
    workers=[ClientWorker(j,x,y,ids) for j,(x,y,ids) in enumerate(blocks)]
    m,_=fedavg(model(ctx.tr['X'].shape[1],ctx.cfg['hidden'],seed),workers,ctx.cfg['fl'],ctx.cfg['fl']['rounds'],seed)
    return m

def checked(root,name,hashes):
    if digest(root/name)!=hashes[str(name)]: raise ValueError(f'Changed checkpoint {root/name}')
    return root/name

def build_requests(ctx):
    tr,te,va=ctx.tr,ctx.data['test'],ctx.data['validation']; base_seed=ctx.cfg['seed']
    base=[(tr['X'][p],tr['y'][p],tr['row_id'][p]) for p in ctx.parts]
    dims=trigger_dims(tr['X'],CONFIG['trigger_count'])
    for split in ['validation','test','calibration']:
        if (ctx.data[split]['X'][:,dims]!=0).any(): raise AssertionError('Trigger dimension occurs naturally')
    triggered=stamp(te['X'][te['y']==1],dims); pool=np.flatnonzero(va['y']==1); requests={}
    for c in CONFIG['canary_clients']:
        cx,cy,src=make_canaries(tr['X'],tr['y'],ctx.parts[c],CONFIG['canary_count'],dims,base_seed+7)
        cids=np.array([f'canary_{i}' for i in range(len(cy))])
        # Held-out canaries: validation attacks stamped and relabelled the same way, never trained on.
        held=np.sort(np.random.default_rng(base_seed+11+c).choice(pool,min(len(cy),len(pool)),replace=False))
        requests[f'canary_c{c}_{len(cy)}']={'kind':'canary','target':c,'primary':'backdoor_p_normal','primary_rows':triggered,
            'original_blocks':[b if j!=c else (np.r_[b[0],cx],np.r_[b[1],cy],np.r_[b[2],cids]) for j,b in enumerate(base)],
            'retained_blocks':base,'forget':(cx,cy,cids),'target_retain':base[c],
            'nonmember':(stamp(va['X'][held],dims),np.zeros(len(held),dtype=np.int64)),
            'audit':{'trigger_dims':dims.tolist(),'source_row_ids':tr['row_id'][src].tolist(),'nonmember_row_ids':va['row_id'][held].tolist()}}
    for case in CONFIG['natural_requests']:
        f,retained,request=ctx.request(case); t=request['target']
        held=class_proportional(tr['y'][f],va['y'],CONFIG['nonmember_cap'],base_seed+13)
        requests[case]={'kind':'natural','target':t,'primary':'mae_F','primary_rows':None,'original_blocks':base,
            'retained_blocks':[(tr['X'][p],tr['y'][p],tr['row_id'][p]) for p in retained],
            'forget':(tr['X'][f],tr['y'][f],tr['row_id'][f]),'target_retain':(tr['X'][retained[t]],tr['y'][retained[t]],tr['row_id'][retained[t]]),
            'nonmember':(va['X'][held],va['y'][held]),'audit':{'nonmember_row_ids':va['row_id'][held].tolist()}}
    return requests

def cloud_models(ctx,name,req,seeds,store,train_missing):
    """Original and retrain clouds. Canary retrains are the clean seed-floor originals."""
    floor=json.loads((SEED_FLOOR/'artifact_hashes.json').read_text())
    from_floor=lambda f:load(ctx,checked(SEED_FLOOR,Path('checkpoints')/f,floor))
    clean=[from_floor(f'original_seed{s}.pt') for s in seeds]
    if req['kind']=='natural': return {'original':clean,'retrain':[from_floor(f'{name}_retrain_seed{s}.pt') for s in seeds]},'seed_floor'
    if (IDENTIFIABLE/'checkpoints'/f'{name}_original_seed{seeds[0]}.pt').exists():
        earlier=json.loads((IDENTIFIABLE/'requests.json').read_text())[name]
        if earlier['source_row_ids']!=req['audit']['source_row_ids'] or earlier['trigger_dims']!=req['audit']['trigger_dims']:
            raise ValueError(f'{name} differs from the identifiable-FU canary')
        ident=json.loads((IDENTIFIABLE/'artifact_hashes.json').read_text())
        return {'original':[load(ctx,checked(IDENTIFIABLE,Path('checkpoints')/f'{name}_original_seed{s}.pt',ident)) for s in seeds],
                'retrain':clean},'identifiable_fu'
    originals=[]
    for s in seeds:
        path=store/'checkpoints'/f'{name}_original_seed{s}.pt'
        if train_missing: m=train(ctx,req['original_blocks'],s); torch.save({'state_dict':state(m),'seed':s},path)
        else: m=load(ctx,path)
        originals.append(m)
    return {'original':originals,'retrain':clean},'this_run'

class Scorer:
    """Request metrics; MAE(F) is to the retrain cloud with the same-seed retrain left out."""
    def __init__(self,ctx,req,retrains):
        self.req=req; self.te=ctx.data['test']; self.fx,self.fy=req['forget'][0],req['forget'][1]
        self.rF=np.array([expit(logit_margin(m,self.fx)) for m in retrains])
    def __call__(self,m,i):
        margin=logit_margin(m,self.fx); mae=float(np.abs(expit(margin)-np.delete(self.rF,i,0)).mean())
        primary=float((1-expit(logit_margin(m,self.req['primary_rows']))).mean()) if self.req['primary']=='backdoor_p_normal' else mae
        return {'primary':primary,'mae_F':mae,'forget_ce':float(-stable_baselines(self.fy,margin)['negative_loss'].mean()),
                'macro_f1_test':float(utility(self.te['y'],expit(logit_margin(m,self.te['X'])))['macro_f1'])}

def gate_entry(values,target,source):
    """Gate on the primary metric: the original and retrain clouds' 95% prediction intervals must not overlap."""
    col=lambda k,key:[v[key] for v in values[k]]
    res=classify(values['original'][0]['primary'],col('retrain','primary'),col('original','primary'))
    mae=classify(values['original'][0]['mae_F'],col('retrain','mae_F'),col('original','mae_F'))
    return {'target':target,'cloud_source':source,'separable':res['separable'],'effect_size':res['separation_effect_size'],
            'retrain_mean':res['retrain_mean'],'retrain_pi':[res['retrain_pi_low'],res['retrain_pi_high']],
            'original_mean':res['original_mean'],'original_pi':[res['original_pi_low'],res['original_pi_high']],
            'mae_F_separable':mae['separable'],'mae_F_effect_size':mae['separation_effect_size']}

def workers_for(req):
    t=req['target']; nx,ny=req['nonmember']
    return (ClientWorker(t,*req['forget']),ClientWorker(t,*req['target_retain']),ClientWorker(t,nx,ny,np.arange(len(ny))),
            [ClientWorker(j,*b) for j,b in enumerate(req['retained_blocks'])])

def stop_levels(orig,fw,nw,retrain_ce,i):
    """Nonmember level: the original's CE on held-out look-alikes. Oracle: retrain-cloud CE(F), same seed left out."""
    return {'nonmember':forget_ce(orig,nw),'retrain_oracle':float(np.delete(retrain_ce,i).mean())}

def run_local(orig,fw,rw,name,levels,s):
    obj=CONFIG['objectives'][name.split('@')[0]]; t0=time.perf_counter()
    pre,_,info=local_forget(orig,fw,rw,{**CONFIG['local'],**obj},s,levels[name.split('@')[1]])
    return pre,info,time.perf_counter()-t0

def run_kd(orig,fw,rw,spec,levels,s,teacher=None):
    """teacher: (model, info, seconds) of the local arm named in spec, when the mode needs one."""
    mode,teacher_arm,stop=spec; t0=time.perf_counter(); tm,tinfo,tsec=teacher if teacher_arm else (None,{},0.)
    pre,_,info=forget_kd(orig,tm,fw,rw,CONFIG['kd'],mode,s,levels[stop] if stop else None)
    info={**info,'teacher_arm':teacher_arm,'kd_steps':info['steps'],'steps':info['steps']+tinfo.get('steps',0),
          'backward_samples':info['backward_samples']+tinfo.get('backward_samples',0),
          'check_samples':info['check_samples']+tinfo.get('check_samples',0)}
    return pre,info,time.perf_counter()-t0+tsec

def kd_name(spec): return spec[0] if spec[2] is None else f'{spec[0]}@{spec[2]}'

def decide(clouds,runs,controls,targets,min_targets):
    rows=[]
    for name,target in targets.items():
        cl=clouds[clouds.request==name]; R=cl[cl.cloud=='retrain'].sort_values('seed'); O=cl[cl.cloud=='original'].sort_values('seed')
        ctrl=controls[controls.request==name].set_index(['stage','seed','control_steps']).sort_index()
        mo,mr=O.primary.mean(),R.primary.mean(); position=lambda v:float(((mo-np.asarray(v))/(mo-mr)).mean())
        for (arm,stage),g in runs[runs.request==name].groupby(['arm','stage']):
            g=g.sort_values('seed')
            if g.seed.tolist()!=O.seed.tolist(): raise ValueError(f'Seed mismatch for {name}/{arm}')
            c=ctrl.loc[[(stage,s,k) for s,k in zip(g.seed,g.control_steps)]]; zero=ctrl.loc[[(stage,s,0) for s in g.seed]]
            pre=runs[(runs.request==name)&(runs.arm==arm)&(runs.stage=='pre')].sort_values('seed')
            gain=paired_gain(g.primary,c.primary,R.primary); base=paired_gain(g.primary,zero.primary,R.primary)
            rows.append({'request':name,'target':target,'arm':arm,'family':g.family.iloc[0],'stage':stage,
                'fu_position':position(g.primary),'control_position':position(c.primary),'retain_only_position':position(zero.primary),
                **gain,'gain_vs_retain_only_mean':base['gain_mean'],'gain_vs_retain_only_ci_low':base['gain_ci_low'],
                'superior_to_retain_only':base['superior'],**utility_noninferior(g.macro_f1_test,c.macro_f1_test,R.macro_f1_test),
                'f1_minus_retrain_mean':float(g.macro_f1_test.mean()-R.macro_f1_test.mean()),
                **local_step_check(O.mae_F,pre.mae_F,R.mae_F),
                'steps_mean':g.steps.mean(),'control_steps_mean':g.control_steps.mean(),'fraction_zero_steps':float((g.steps==0).mean()),
                'fraction_stopped_target':float(g.stop_reason.isin(['target','ascent_target']).mean()),
                'fraction_stopped_ceiling':float(g.stop_reason.isin(['ce_ceiling','ascent_ce_ceiling']).mean()),
                'local_seconds_mean':g.local_seconds.mean()})
    decision=pd.DataFrame(rows)
    verdict={f'{arm}/{stage}':overall(g.to_dict('records'),min_targets) for (arm,stage),g in decision.groupby(['arm','stage'])} if len(decision) else {}
    return decision,verdict

def run(output,seeds_limit=None):
    ctx=Context(); started=time.perf_counter()
    seeds=json.loads((SEED_FLOOR/'manifest.json').read_text())['protocol']['seeds'][:seeds_limit]
    print(f'Using device: {get_device()}',flush=True)
    ctx.manifest_for(output,'forget_signal',{**CONFIG,'seeds':seeds,'seed_floor_source':str(SEED_FLOOR),'identifiable_source':str(IDENTIFIABLE),
        'gate':'request enters only if the 95% prediction intervals of its original and retrain clouds do not overlap on the primary metric',
        'control_rule':'per arm and seed: Adam on CE(R_target) for ceil(backward_samples/batch) steps, plus 0-step retain_only; same recovery',
        'decision':'superior iff lower 95% CI of paired gain (|control-mu_R|-|fu-mu_R|) > retrain-cloud PI half-width; '
                   'pass iff superior, F1-noninferior and local MAE(F) reduced on every admitted request spanning >= min_targets targets',
        'stop_levels':'nonmember = original CE on held-out look-alikes of F; retrain_oracle = retrain-cloud mean CE(F), same seed left out (oracle, not deployable)',
        'recovery_rounds':ctx.cfg['recovery_rounds'],'selection':'none; every arm reported'})
    (output/'checkpoints').mkdir()
    requests=build_requests(ctx)
    write_json(output/'requests.json',{k:{'kind':v['kind'],'target':v['target'],'primary':v['primary'],'forget_count':len(v['forget'][1]),
        'nonmember_count':len(v['nonmember'][1]),**v['audit']} for k,v in requests.items()})
    cloud_rows,gate,runs,controls,levels_rows=[],{},[],[],[]
    for name,req in requests.items():
        clouds,source=cloud_models(ctx,name,req,seeds,output,True); score=Scorer(ctx,req,clouds['retrain'])
        values={k:[score(m,i) for i,m in enumerate(ms)] for k,ms in clouds.items()}
        cloud_rows.extend({'request':name,'cloud':k,'seed':s,**v} for k,vals in values.items() for s,v in zip(seeds,vals))
        gate[name]=gate_entry(values,req['target'],source)
        print(f'{name}: gate separable={gate[name]["separable"]} effect={gate[name]["effect_size"]:.2f}',flush=True)
        if not gate[name]['separable']: continue
        fw,rw,nw,recovery=workers_for(req); retrain_ce=np.array([forget_ce(m,fw) for m in clouds['retrain']])
        for i,(s,orig) in enumerate(zip(seeds,clouds['original'])):
            levels=stop_levels(orig,fw,nw,retrain_ce,i)
            levels_rows.append({'request':name,'seed':s,'original_forget_ce':forget_ce(orig,fw),**{f'{k}_ce':v for k,v in levels.items()}})
            recover=lambda pre:fedavg(pre,recovery,ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s+900000)[0]
            done=set()
            def control(steps):
                if steps in done: return
                t0=time.perf_counter(); pre=orig if steps==0 else retain_matched(orig,rw,CONTROL,steps,s)[0]
                local=time.perf_counter()-t0; post=recover(pre); done.add(steps)
                for stage,m in [('pre',pre),('post',post)]:
                    controls.append({'request':name,'seed':s,'control_steps':steps,'stage':stage,'local_seconds':local,**score(m,i)})
            def record(arm,family,pre,info,local_seconds):
                steps=matched_steps(info['backward_samples'],CONTROL['batch_size']); control(steps)
                t0=time.perf_counter(); post=recover(pre)
                common={'request':name,'target':req['target'],'seed':s,'arm':arm,'family':family,'control_steps':steps,
                        'local_seconds':local_seconds,'recovery_seconds':time.perf_counter()-t0,**info}
                for stage,m in [('pre',pre),('post',post)]: runs.append({**common,'stage':stage,**score(m,i)})
            control(0); local_models={}
            for obj in CONFIG['objectives']:
                for stop in CONFIG['stops']:
                    arm=f'{obj}@{stop}'; local_models[arm]=run_local(orig,fw,rw,arm,levels,s); record(arm,'local',*local_models[arm])
            for spec in CONFIG['kd_arms']:
                record(kd_name(spec),'kd',*run_kd(orig,fw,rw,spec,levels,s,local_models.get(spec[1])))
            print(f'{name}: seed {i+1}/{len(seeds)} ({time.perf_counter()-started:.0f}s)',flush=True)
    clouds=pd.DataFrame(cloud_rows); runs=pd.DataFrame(runs); controls=pd.DataFrame(controls)
    clouds.to_csv(output/'clouds.csv',index=False); runs.to_csv(output/'fu_runs.csv',index=False)
    controls.to_csv(output/'controls.csv',index=False); pd.DataFrame(levels_rows).to_csv(output/'stop_levels.csv',index=False)
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
