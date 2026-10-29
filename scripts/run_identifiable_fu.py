"""Identifiable deletion benchmark: canary and attack-type requests, seed-cloud gate, stronger forget grid."""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.special import expit
from scipy.stats import t as t_dist
from scripts.experiment_context import Context,finalize
from src.data.canary import trigger_dims,stamp,make_canaries
from src.data.prepare import digest,write_json
from src.eval.membership_validation import logit_margin,stable_baselines
from src.eval.metrics import utility
from src.eval.seed_floor import classify
from src.fl.fedavg_core import model,state,fedavg,ClientWorker,get_device
from src.fu.strong_forget import ga_adam,neggrad_plus

SEED_FLOOR=Path('runs/seed_floor_seed42')
CONFIG={'canary_client':3,'canary_count':500,'trigger_count':3,'type_requests':[[4,'mitm'],[4,'xss']],
        'forget':{'lr':1e-3,'batch_size':128,'clip':5.,'ce_margin':10.,'alpha':.9},
        'steps':[3,10,30,100],'methods':['ga_adam','neggrad_plus']}

class Blocks:
    """Per-client arrays; FedAvg workers are built from these."""
    def __init__(self,blocks): self.blocks=blocks
    def workers(self): return [ClientWorker(j,x,y,ids) for j,(x,y,ids) in enumerate(self.blocks)]

def train(ctx,blocks,seed):
    m,_=fedavg(model(ctx.tr['X'].shape[1],ctx.cfg['hidden'],seed),blocks.workers(),ctx.cfg['fl'],ctx.cfg['fl']['rounds'],seed)
    return m

def load(ctx,path):
    ck=torch.load(path,weights_only=True); m=model(ctx.tr['X'].shape[1],ctx.cfg['hidden'],ck['seed'])
    m.load_state_dict(ck['state_dict']); return m

def mean_ci(v):
    v=np.asarray(v,dtype=float); v=v[np.isfinite(v)]
    if len(v)<2: return float(v.mean()) if len(v) else float('nan'),float('nan'),float('nan')
    h=t_dist.ppf(.975,len(v)-1)*v.std(ddof=1)/np.sqrt(len(v)); return float(v.mean()),float(v.mean()-h),float(v.mean()+h)

def run(output,seeds_limit=None):
    ctx=Context(); tr,te=ctx.tr,ctx.data['test']; base_seed=ctx.cfg['seed']; started=time.perf_counter()
    floor_hashes=json.loads((SEED_FLOOR/'artifact_hashes.json').read_text())
    seeds=json.loads((SEED_FLOOR/'manifest.json').read_text())['protocol']['seeds'][:seeds_limit]
    for s in seeds:
        name=str(Path('checkpoints')/f'original_seed{s}.pt')
        if digest(SEED_FLOOR/name)!=floor_hashes[name]: raise ValueError(f'Changed seed-floor checkpoint {name}')
    print(f'Using device: {get_device()}',flush=True)
    ctx.manifest_for(output,'identifiable_fu',{**CONFIG,'seeds':seeds,'seed_floor_source':str(SEED_FLOOR),
        'gate':'request enters FU comparison only if 95% prediction intervals of original and retrain clouds do not overlap on its primary metric',
        'fu':'applied to each original seed; retain_only plus every method x steps; same 2 retained recovery rounds (seed+900000)',
        'selection':'none; full grid reported','recovery_rounds':ctx.cfg['recovery_rounds']})
    (output/'checkpoints').mkdir()
    base=[(tr['X'][p],tr['y'][p],tr['row_id'][p]) for p in ctx.parts]
    dims=trigger_dims(tr['X'],CONFIG['trigger_count'])
    for split in ['validation','test','calibration']:
        if (ctx.data[split]['X'][:,dims]!=0).any(): raise AssertionError('Trigger dimension occurs naturally')
    c=CONFIG['canary_client']
    cx,cy,src=make_canaries(tr['X'],tr['y'],ctx.parts[c],CONFIG['canary_count'],dims,base_seed+7)
    cids=np.array([f'canary_{i}' for i in range(len(cy))])
    triggered=stamp(te['X'][te['y']==1],dims)
    clean_originals=[load(ctx,SEED_FLOOR/'checkpoints'/f'original_seed{s}.pt') for s in seeds]
    requests={'canary_c3_500':{'original':Blocks([b if j!=c else (np.r_[b[0],cx],np.r_[b[1],cy],np.r_[b[2],cids]) for j,b in enumerate(base)]),
                               'retained':Blocks(base),'forget':(cx,cy,cids),'target':c,'target_retain':base[c],
                               'primary':'backdoor_p_normal','primary_rows':triggered,'primary_direction':'P(normal) on trigger-stamped test attacks',
                               'clean_cloud':'retrain','audit':{'trigger_dims':dims.tolist(),'source_row_ids':tr['row_id'][src].tolist()}}}
    for target,family in CONFIG['type_requests']:
        local=ctx.parts[target]; f=local[ctx.types['train'][local]==family]
        retained=[np.setdiff1d(p,f) for p in ctx.parts]
        requests[f'type_c{target}_{family}']={'original':Blocks(base),'retained':Blocks([(tr['X'][p],tr['y'][p],tr['row_id'][p]) for p in retained]),
            'forget':(tr['X'][f],tr['y'][f],tr['row_id'][f]),'target':target,
            'target_retain':(tr['X'][retained[target]],tr['y'][retained[target]],tr['row_id'][retained[target]]),
            'primary':'type_p_attack','primary_rows':te['X'][ctx.types['test']==family],'primary_direction':f'P(attack) on test {family} rows',
            'clean_cloud':'original','audit':{'family':family,'forget_count':len(f),'family_rows_elsewhere':int((ctx.types['train']==family).sum()-len(f)),
                                             'test_rows':int((ctx.types['test']==family).sum()),'forget_row_ids':tr['row_id'][f].tolist()}}
    write_json(output/'requests.json',{k:{'primary':v['primary'],'primary_direction':v['primary_direction'],'target':v['target'],
        'forget_count':len(v['forget'][1]),**v['audit']} for k,v in requests.items()})
    def evaluate(m,req):
        fx,fy,_=req['forget']; p=expit(logit_margin(m,req['primary_rows']))
        return {'primary':float((1-p).mean() if req['primary']=='backdoor_p_normal' else p.mean()),
                'forget_ce':float(-stable_baselines(fy,logit_margin(m,fx))['negative_loss'].mean()),
                'macro_f1_test':float(utility(te['y'],expit(logit_margin(m,te['X'])))['macro_f1'])}
    clouds=[]; gate={}; runs=[]
    for name,req in requests.items():
        cloud_models={}
        for kind in ['original','retrain']:
            if kind==req['clean_cloud']: cloud_models[kind]=clean_originals; continue
            blocks=req['original'] if kind=='original' else req['retained']; cloud_models[kind]=[]
            for s in seeds:
                m=train(ctx,blocks,s); cloud_models[kind].append(m)
                torch.save({'state_dict':state(m),'seed':s},output/'checkpoints'/f'{name}_{kind}_seed{s}.pt')
        values={kind:[evaluate(m,req) for m in ms] for kind,ms in cloud_models.items()}
        for kind,vals in values.items():
            clouds.extend({'request':name,'cloud':kind,'seed':s,**v} for s,v in zip(seeds,vals))
        r=[v['primary'] for v in values['retrain']]; o=[v['primary'] for v in values['original']]
        res=classify(o[0],r,o); gate[name]={'separable':res['separable'],'effect_size':res['separation_effect_size'],
            'retrain_mean':res['retrain_mean'],'retrain_pi':[res['retrain_pi_low'],res['retrain_pi_high']],
            'original_mean':res['original_mean'],'original_pi':[res['original_pi_low'],res['original_pi_high']]}
        print(f'{name}: gate separable={res["separable"]} effect={res["separation_effect_size"]:.2f}',flush=True)
        if not res['separable']: continue
        fw=ClientWorker(req['target'],*req['forget']); rw=ClientWorker(req['target'],*req['target_retain'])
        recovery_workers=req['retained'].workers()
        for i,(s,orig) in enumerate(zip(seeds,cloud_models['original'])):
            arms=[('retain_only',0,None)]+[(mth,k,None) for mth in CONFIG['methods'] for k in CONFIG['steps']]
            for method,steps,_ in arms:
                t0=time.perf_counter()
                if method=='retain_only': pre,logs=orig,[]
                elif method=='ga_adam': pre,logs=ga_adam(orig,fw,CONFIG['forget'],steps,s)
                else: pre,logs=neggrad_plus(orig,fw,rw,CONFIG['forget'],steps,s)
                post,_=fedavg(pre,recovery_workers,ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s+900000)
                common={'request':name,'seed':s,'method':method,'steps':steps,
                        'realized_steps':sum(r.get('optimizer_steps',0) for r in logs),
                        'stopped_at_ceiling':any(r.get('stopped')=='ce_ceiling' for r in logs),'seconds':time.perf_counter()-t0}
                for stage,m in [('pre',pre),('post',post)]:
                    runs.append({**common,'stage':stage,**evaluate(m,req)})
            print(f'{name}: seed {i+1}/{len(seeds)}',flush=True)
    clouds=pd.DataFrame(clouds); runs=pd.DataFrame(runs)
    clouds.to_csv(output/'clouds.csv',index=False); runs.to_csv(output/'fu_runs.csv',index=False); write_json(output/'gate.json',gate)
    summary=[]
    for (name,method,steps,stage),g in (runs.groupby(['request','method','steps','stage']) if len(runs) else []):
        cl=clouds[clouds.request==name]; r=cl[cl.cloud=='retrain']; o=cl[cl.cloud=='original']
        placed=[classify(v,r.primary,o.primary) for v in g.primary]
        f1=[classify(v,r.macro_f1_test,o.macro_f1_test) for v in g.macro_f1_test]
        pos=mean_ci([p['forgetting_position'] for p in placed]); df1=mean_ci(g.macro_f1_test-r.macro_f1_test.mean())
        summary.append({'request':name,'method':method,'steps':steps,'stage':stage,'seeds':len(g),
            'primary_mean':g.primary.mean(),'forgetting_position':pos[0],'position_ci_low':pos[1],'position_ci_high':pos[2],
            'fraction_retrain_like':np.mean([p['verdict']=='RETRAIN_LIKE' for p in placed]),
            'fraction_beyond_retrain':np.mean([p['verdict']=='BEYOND_RETRAIN' for p in placed]),
            'f1_minus_retrain_mean':df1[0],'f1_ci_low':df1[1],'f1_ci_high':df1[2],
            'fraction_f1_below_retrain_pi':np.mean([v<p['retrain_pi_low'] for v,p in zip(g.macro_f1_test,f1)]),
            'forget_ce_mean':g.forget_ce.mean(),'realized_steps_mean':g.realized_steps.mean(),
            'fraction_stopped_at_ceiling':g.stopped_at_ceiling.mean(),'seconds_mean':g.seconds.mean()})
    summary=pd.DataFrame(summary); summary.to_csv(output/'summary.csv',index=False)
    for name in summary.request.unique() if len(summary) else []:
        s=summary[summary.request==name]; fig,ax=plt.subplots(figsize=(7,5))
        for (method,stage),g in s.groupby(['method','stage']):
            g=g.sort_values('steps'); fill='full' if stage=='post' else 'none'
            x,y=g.forgetting_position.to_numpy(),g.f1_minus_retrain_mean.to_numpy()
            ax.errorbar(x,y,xerr=np.abs([x-g.position_ci_low.to_numpy(),g.position_ci_high.to_numpy()-x]),
                        marker='o',fillstyle=fill,label=f'{method}/{stage}',capsize=2)
            for _,row in g.iterrows(): ax.annotate(str(int(row.steps)),(row.forgetting_position,row.f1_minus_retrain_mean),fontsize=6)
        ax.axvline(0,color='grey',ls=':'); ax.axvline(1,color='k',ls='--'); ax.axhline(0,color='k',ls='--',lw=.8)
        ax.set(xlabel='forgetting position (0 = original cloud, 1 = retrain cloud)',ylabel='test macro-F1 minus retrain-cloud mean',title=name)
        ax.legend(fontsize=7); fig.tight_layout(); fig.savefig(output/f'{name}_tradeoff.png',dpi=160); plt.close(fig)
    write_json(output/'status.json',{'completed':True,'seeds':len(seeds),'requests':list(requests),
        'requests_passing_gate':[k for k,v in gate.items() if v['separable']],'fu_rows':len(runs),
        'total_seconds':time.perf_counter()-started,'selection_performed':False,'unlearning_success_claimed':False})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=None,help='use only the first N seed-floor seeds (smoke runs)')
    a=p.parse_args(); run(a.output,a.seeds)
