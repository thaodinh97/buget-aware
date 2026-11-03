"""Confirmation and utility measurements for the frozen deployable method (protocol: docs/confirmation_plan.md).

--mode existing  The 5 requests, seeds and clouds of forget_signal_seed42 / deployable_forget_seed42, plus
                 combination-trigger canaries on the same clients and source rows (their original clouds are
                 trained here). Adds per-client utility and FPR, the pool-matched control and the reset
                 baselines to the reported results, and cross-checks the frozen arms against the stored run.
--mode confirm   A new Dirichlet partition and new seeds; natural requests on every client, zero-dim and
                 combination canaries; every cloud is trained here.
--mode confirm2  The second confirmation (CONFIRM2): as confirm on another partition and other seeds, with the
                 utility floor never finer than one row. Fixed in docs/confirmation_plan.md after confirmation_p43
                 and before this run; there is no third confirmation.
Arms, controls and the decision rule are frozen in CONFIG for every mode; nothing is selected from results.
"""
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
from scripts import run_forget_signal as R, run_deployable_forget as D
from src.data.canary import trigger_dims,stamp,make_canaries,combination_trigger,stamp_combination,make_combination_canaries
from src.data.federated import partition,ids_hash
from src.data.prepare import digest,write_json
from src.data.request_sensitivity import build_request
from src.eval.confirmation_decision import request_row,overall,one_row_resolution
from src.eval.membership_validation import logit_margin
from src.eval.utility_breakdown import detector_metrics,client_metrics
from src.fl.fedavg_core import model,state,fedavg,ClientWorker,get_device
from src.fu.bounded_forget import local_forget,retain_matched_many,forget_ce,forget_prob
from src.fu.neighbor_reference import knn_indices
from src.fu.reset_baseline import dead_dims,reset_input_columns

DEPLOYABLE=Path('runs/deployable_forget_seed42')
CONFIG={'arms':['ga_proj@joint_aux_prob','nkd_proj@joint_aux_prob','ga_adam@joint_aux_prob','ga_proj@retrain_oracle'],
        'recommended_arm':'ga_proj@joint_aux_prob','oracle_arms':['ga_proj@retrain_oracle'],
        'resets':['reset_init@local','reset_init@global','reset_zero@global'],
        'local':D.CONFIG['local'],'ops':{k:D.CONFIG['ops'][k] for k in ['ga_proj','nkd_proj','ga_adam']},
        'knn_k':D.CONFIG['knn_k'],'pools':D.CONFIG['pools'],'reference':D.CONFIG['reference'],
        'control':R.CONTROL,'control_pool':['target_retain','validation'],
        'control_round_above':D.CONFIG['control_round_above'],'control_round':D.CONFIG['control_round'],
        'canary_count':500,'canary_seed_offset':7,'zero_trigger_count':3,'combination_features':['proto','service','conn_state'],
        'existing':{'canary_clients':[3,0,4]},
        'confirm':{'partition_seed':43,'alpha':.5,'clients':5,'seeds':[20042+i for i in range(10)],
                   'designs':['random_stratified','low_overlap_groups'],'fraction':.5,'request_seed_offset':1,
                   'canary_min_attack_rows':2000,'max_canary_clients':3},
        'level':.95,'min_targets':2}
# Kept out of CONFIG so that the runs made before it still verify against their recorded protocol.
CONFIRM2={**CONFIG['confirm'],'partition_seed':44,'seeds':[30042+i for i in range(10)],'utility_floor':'one_row'}
METRICS=['primary','mae_F','test_macro_f1','test_fpr']

def confirm_config(mode): return CONFIRM2 if mode=='confirm2' else CONFIG['confirm']

def one_row_floor(mode): return mode!='existing' and confirm_config(mode).get('utility_floor')=='one_row'

class UtilityScorer(R.Scorer):
    """R.Scorer plus test FPR/recall/class F1 and the same metrics on every client's retained rows."""
    def __init__(self,ctx,req,retrains):
        super().__init__(ctx,req,retrains); self.blocks=[(np.asarray(b[0]),np.asarray(b[1])) for b in req['retained_blocks']]
    def __call__(self,m,i):
        out=super().__call__(m,i)
        out.update({f'test_{k}':v for k,v in detector_metrics(self.te['y'],expit(logit_margin(m,self.te['X']))).items()})
        out.update(client_metrics([expit(logit_margin(m,x)) for x,_ in self.blocks],[y for _,y in self.blocks]))
        return out

def seeds_for(mode,limit=None):
    seeds=json.loads((R.SEED_FLOOR/'manifest.json').read_text())['protocol']['seeds'] if mode=='existing' else confirm_config(mode)['seeds']
    return list(seeds)[:limit]

def setting(ctx,mode):
    """Client partition (train indices) and the base seed of the canaries and requests."""
    if mode=='existing': return ctx.parts,ctx.cfg['seed']
    c=confirm_config(mode); return partition(ctx.tr['y'],c['partition_seed'],c['alpha'],c['clients']),c['partition_seed']

def blocks_of(tr,parts): return [(tr['X'][p],tr['y'][p],tr['row_id'][p]) for p in parts]

def triggers(ctx):
    tr=ctx.tr; names=json.loads((ctx.data_path/'transformer.json').read_text())['feature_names']
    if len(names)!=tr['X'].shape[1]: raise ValueError('Feature names do not match the encoded data')
    splits=[ctx.data[s]['X'] for s in ['train','validation','calibration','test']]
    dims=trigger_dims(tr['X'],CONFIG['zero_trigger_count'])
    if any((x[:,dims]!=0).any() for x in splits): raise AssertionError('Trigger dimension occurs naturally')
    return {'zero':{'dims':dims.tolist(),'names':[names[d] for d in dims]},
            'combination':combination_trigger(splits,tr['X'],names,CONFIG['combination_features'])}

def canary_request(ctx,base,parts,c,kind,trig,seed):
    tr,te=ctx.tr,ctx.data['test']; n=CONFIG['canary_count']; attacks=te['X'][te['y']==1]
    if kind=='zero':
        dims=np.array(trig['dims']); cx,cy,src=make_canaries(tr['X'],tr['y'],parts[c],n,dims,seed); rows=stamp(attacks,dims)
    else:
        cx,cy,src=make_combination_canaries(tr['X'],tr['y'],parts[c],n,trig,seed); rows=stamp_combination(attacks,trig)
    cids=np.array([f'canary_{i}' for i in range(n)])
    return {'kind':'canary','trigger':kind,'target':c,'primary':'backdoor_p_normal','primary_rows':rows,
            'original_blocks':[b if j!=c else (np.r_[b[0],cx],np.r_[b[1],cy],np.r_[b[2],cids]) for j,b in enumerate(base)],
            'retained_blocks':base,'forget':(cx,cy,cids),'target_retain':base[c],'audit':{'source_row_ids':tr['row_id'][src].tolist()}}

def natural_request(ctx,base,parts,t,design,seed,signatures,fraction):
    tr=ctx.tr; f,r,audit=build_request(tr,parts,t,fraction,design,signatures,seed)
    keep=(tr['X'][r],tr['y'][r],tr['row_id'][r])
    return {'kind':'natural','trigger':None,'target':t,'primary':'mae_F','primary_rows':None,'original_blocks':base,
            'retained_blocks':[b if j!=t else keep for j,b in enumerate(base)],
            'forget':(tr['X'][f],tr['y'][f],tr['row_id'][f]),'target_retain':keep,
            'audit':{**audit,'forget_hash':ids_hash(tr['row_id'][f])}}

def make_requests(ctx,mode):
    """Requests, in a fixed order, from metadata only; invalid natural requests are reported, never resampled."""
    parts,seed=setting(ctx,mode); base=blocks_of(ctx.tr,parts); trig=triggers(ctx); n=CONFIG['canary_count']
    off=CONFIG['canary_seed_offset']; requests,invalid={},{}
    if mode=='existing':
        requests.update(R.build_requests(ctx))
        for c in CONFIG['existing']['canary_clients']:
            req=canary_request(ctx,base,parts,c,'combination',trig['combination'],seed+off)
            if req['audit']['source_row_ids']!=requests[f'canary_c{c}_{n}']['audit']['source_row_ids']:
                raise AssertionError('Combination canaries must reuse the zero-dim canaries\' source rows')
            requests[f'combo_canary_c{c}_{n}']=req
        for k,v in requests.items(): v.setdefault('trigger','zero' if v['kind']=='canary' else None)
        return parts,trig,requests,invalid,[]
    c=confirm_config(mode); y=ctx.tr['y']; attack=[int((y[p]==1).sum()) for p in parts]
    canary_clients=[j for j in sorted(range(len(parts)),key=lambda j:(-attack[j],j)) if attack[j]>=c['canary_min_attack_rows']][:c['max_canary_clients']]
    for kind,prefix in [('zero','canary'),('combination','combo_canary')]:
        for j in canary_clients: requests[f'{prefix}_c{j}_{n}']=canary_request(ctx,base,parts,j,kind,trig[kind],seed+off)
    signatures=ctx.metadata.loc[ctx.tr['row_id'],'feature_signature'].to_numpy()
    for t in range(len(parts)):
        for design in c['designs']:
            name=f'client{t}_{design}_{c["fraction"]}'
            try: requests[name]=natural_request(ctx,base,parts,t,design,seed+c['request_seed_offset'],signatures,c['fraction'])
            except ValueError as exc: invalid[name]=str(exc)
    return parts,trig,requests,invalid,canary_clients

def load_or_train(ctx,store,fname,blocks,s,train):
    path=store/'checkpoints'/fname
    if not train: return R.load(ctx,path)
    m=R.train(ctx,blocks,s); torch.save({'state_dict':state(m),'seed':s},path); return m

def clean_cloud(ctx,mode,base,seeds,store,train):
    """Originals on the clean partition: the natural requests' original cloud and the canaries' retrain cloud."""
    if mode=='existing':
        floor=json.loads((R.SEED_FLOOR/'artifact_hashes.json').read_text())
        return [R.load(ctx,R.checked(R.SEED_FLOOR,Path('checkpoints')/f'original_seed{s}.pt',floor)) for s in seeds]
    return [load_or_train(ctx,store,f'original_seed{s}.pt',base,s,train) for s in seeds]

def clouds_for(ctx,mode,name,req,seeds,clean,store,train):
    if mode=='existing' and not name.startswith('combo_'):
        return R.cloud_models(ctx,name,req,seeds,D.FORGET_SIGNAL,False)
    if req['kind']=='canary':
        return {'original':[load_or_train(ctx,store,f'{name}_original_seed{s}.pt',req['original_blocks'],s,train) for s in seeds],
                'retrain':clean},'this_run'
    return {'original':clean,'retrain':[load_or_train(ctx,store,f'{name}_retrain_seed{s}.pt',req['retained_blocks'],s,train) for s in seeds]},'this_run'

def pool_worker(ctx,req):
    parts=D.parts_for(ctx,req); keys=CONFIG['control_pool']
    x=np.concatenate([parts[k][0] for k in keys]); y=np.concatenate([parts[k][1] for k in keys])
    return ClientWorker(req['target'],x,y,np.arange(len(y)))

class Request:
    """Per-request state shared by all seeds: workers, kNN neighbours, retrain-cloud stop statistics, dead dims."""
    def __init__(self,ctx,req,clouds):
        t=req['target']; self.ctx,self.req=ctx,req; self.fx,self.fy=req['forget'][0],req['forget'][1]
        self.fw=ClientWorker(t,*req['forget']); self.rw=ClientWorker(t,*req['target_retain']); self.pw=pool_worker(ctx,req)
        self.recovery=[ClientWorker(j,*b) for j,b in enumerate(req['retained_blocks'])]
        self.pools=D.pools_for(ctx,req); t0=time.perf_counter()
        self.neighbors={k:knn_indices(self.fx,pool,CONFIG['knn_k']) for k,pool in self.pools.items()}
        self.knn_seconds=time.perf_counter()-t0
        self.rce,self.rp=(np.array([f(m,self.fw) for m in clouds['retrain']]) for f in (forget_ce,forget_prob))
        self.dims={'local':dead_dims(req['target_retain'][0]),'global':dead_dims(*[b[0] for b in req['retained_blocks']])}

    def recover(self,pre,s):
        t0=time.perf_counter(); post,_=fedavg(pre,self.recovery,self.ctx.cfg['fl'],self.ctx.cfg['recovery_rounds'],s+900000)
        return post,time.perf_counter()-t0

    def seed_models(self,orig,s,i):
        """[(arm, family, pre, post, info)] for every frozen arm and reset baseline of one original seed."""
        ref,ref_rows,ref_seconds=D.reference_model(self.ctx,self.req,s)
        _,levels,targets=D.references(orig,self.fx,self.fy,self.pools,self.neighbors,self.rce,self.rp,i,self.fw.device,ref)
        out=[]
        for arm in CONFIG['arms']:
            op,stop=arm.split('@')
            cfg={**CONFIG['local'],**CONFIG['ops'][op],'stop_statistic':'prob' if stop.endswith('_prob') else 'ce'}
            tg=targets[stop.removesuffix('_prob')] if cfg['objective']=='soft_target' else None
            t0=time.perf_counter(); pre,_,info=local_forget(orig,self.fw,self.rw,cfg,s,levels[stop],tg); local=time.perf_counter()-t0
            extra,extra_s=(ref_rows,ref_seconds) if D.charged(stop) else (0,0.)
            info={**info,'backward_samples':info['backward_samples']+extra,'reference_backward_samples':extra}
            post,rec=self.recover(pre,s); out.append((arm,'local',pre,post,{**info,'local_seconds':local+extra_s,'recovery_seconds':rec}))
        init=model(self.fx.shape[1],self.ctx.cfg['hidden'],s)
        for arm in CONFIG['resets']:
            kind,scope=arm.split('@'); dims=self.dims[scope]
            t0=time.perf_counter(); pre=reset_input_columns(orig,init,dims,kind.removeprefix('reset_')); local=time.perf_counter()-t0
            post,rec=self.recover(pre,s)
            out.append((arm,'reset',pre,post,{'steps':0,'stop_reason':'reset','backward_samples':0,'check_samples':0,
                        'reset_dims':'/'.join(map(str,dims)),'reset_dim_count':len(dims),'local_seconds':local,'recovery_seconds':rec}))
        return out

    def controls(self,orig,s,budgets):
        """0-step retain-only, then compute-matched Adam on R_target ('target') and on the reference pool ('pool')."""
        out=[]
        for kind,worker in [('target',self.rw),('pool',self.pw)]:
            t0=time.perf_counter(); matched=retain_matched_many(orig,worker,CONFIG['control'],budgets,s); one=time.perf_counter()-t0
            for steps,pre in sorted(matched.items()):
                if steps==0 and kind=='pool': continue
                post,rec=self.recover(pre,s)
                out.append(('retain_only' if steps==0 else kind,steps,pre,post,one*steps/max(budgets) if steps else 0.,rec))
        return out

def resolutions_of(rc,clients):
    """One-row resolution of every utility check at the retrain cloud's confusion matrices (the confirm2 floor rule)."""
    res=lambda metric,p:one_row_resolution(metric,*(rc[f'{p}_{k}'] for k in ['normal_rows','attack_rows','fpr','recall_attack']))
    return {'test_macro_f1':res('macro_f1','test'),'test_fpr':res('fpr','test'),
            **{f'client{j}_macro_f1':res('macro_f1',f'client{j}') for j in clients}}

def decide(clouds,runs,controls,admitted,clients,one_row=False):
    """Decision rows for every (request, arm, stage) and the per-arm verdict over post-recovery rows.
    one_row: the confirm2 rule, under which no utility floor is finer than one row."""
    cols=METRICS+[f'client{j}_macro_f1' for j in clients]; rows=[]
    arrays=lambda df:{c:df[c].to_numpy(np.float64) for c in cols}
    for name,target in admitted.items():
        cl=clouds[clouds.request==name]; rc=cl[cl.cloud=='retrain'].sort_values('seed'); oc=cl[cl.cloud=='original'].sort_values('seed')
        resolutions=resolutions_of(rc,clients) if one_row else None
        ctl=controls[controls.request==name].set_index(['kind','stage','seed','control_steps']).sort_index()
        fu=runs[runs.request==name]
        for (arm,stage),g in fu.groupby(['arm','stage'],sort=False):
            g=g.sort_values('seed'); seeds=g.seed.tolist()
            if seeds!=oc.seed.tolist() or seeds!=rc.seed.tolist(): raise ValueError(f'Seed mismatch for {name}/{arm}')
            pick=lambda kind,steps:ctl.loc[[('retain_only' if k==0 else kind,stage,s,int(k)) for s,k in zip(seeds,steps)]]
            pre=fu[(fu.arm==arm)&(fu.stage=='pre')].sort_values('seed')
            row=request_row(arrays(g),arrays(pick('target',[0]*len(seeds))),arrays(pick('target',g.control_steps)),
                            arrays(pick('pool',g.control_steps)),arrays(rc),arrays(oc),arrays(pre),clients,CONFIG['level'],resolutions)
            rows.append({'request':name,'target':target,'arm':arm,'family':g.family.iloc[0],'stage':stage,**row,
                         'steps_mean':float(g.steps.mean()),'control_steps_mean':float(g.control_steps.mean()),
                         'local_seconds_mean':float(g.local_seconds.mean())})
    decision=pd.DataFrame(rows)
    post=decision[decision.stage=='post'] if len(decision) else decision
    verdict={arm:overall(g.to_dict('records'),CONFIG['min_targets']) for arm,g in post.groupby('arm',sort=False)} if len(post) else {}
    return decision,verdict

def crosscheck(runs,controls):
    """Existing mode: the frozen arms and R_target controls must reproduce deployable_forget_seed42 (standard recovery)."""
    if not (DEPLOYABLE/'fu_runs.csv').exists(): return {'available':False}
    old,oc=pd.read_csv(DEPLOYABLE/'fu_runs.csv'),pd.read_csv(DEPLOYABLE/'controls.csv'); diffs,n,steps_equal=[0.],0,True
    for arm in CONFIG['arms']:
        m=runs[runs.arm==arm].merge(old[old.arm==arm],on=['request','seed','stage'],suffixes=('','_old'))
        for c in ['primary','mae_F','macro_f1_test']: diffs.extend(np.abs(m[c]-m[f'{c}_old']).tolist())
        steps_equal&=bool((m.steps==m.steps_old).all() and (m.control_steps==m.control_steps_old).all()); n+=len(m)
    c=controls[controls.kind.isin(['target','retain_only'])].merge(oc,on=['request','seed','control_steps','stage'],suffixes=('','_old'))
    for col in ['primary','mae_F','macro_f1_test']: diffs.extend(np.abs(c[col]-c[f'{col}_old']).tolist())
    return {'available':True,'fu_rows':n,'control_rows':len(c),'max_abs_diff':float(max(diffs)),'steps_equal':steps_equal,
            'matches':bool(max(diffs)<=1e-9 and steps_equal)}

def run(output,mode,seeds_limit=None,only=None,ctx=None):
    ctx=ctx or Context(); started=time.perf_counter(); seeds=seeds_for(mode,seeds_limit)
    if mode=='existing': D.check_source()
    print(f'Using device: {get_device()}',flush=True)
    parts,trig,requests,invalid,canary_clients=make_requests(ctx,mode)
    if only: requests={k:v for k,v in requests.items() if k in only}
    ctx.manifest_for(output,f'confirmation_{mode}',{**CONFIG,'mode':mode,'seeds':seeds,'only_requests':only,
        'triggers':trig,'canary_clients':canary_clients if mode!='existing' else CONFIG['existing']['canary_clients'],
        'forget_signal_artifact_hashes_sha256':digest(D.FORGET_SIGNAL/'artifact_hashes.json') if mode=='existing' else None,
        'decision_rule':'src/eval/confirmation_decision.py (docs/confirmation_plan.md); legacy forget_signal rule reported alongside',
        **({'confirm2':CONFIRM2,'utility_floor':'max(retrain-cloud noise floor, one-row resolution at the retrain cloud)'}
           if mode=='confirm2' else {}),
        'selection':'none; every arm, baseline and request reported'})
    (output/'checkpoints').mkdir()
    if mode!='existing':
        tr=ctx.tr; np.savez_compressed(output/'partitions.npz',**{f'client_{i}':tr['row_id'][p] for i,p in enumerate(parts)})
        write_json(output/'clients.json',[{'client':i,'normal':int((tr['y'][p]==0).sum()),'attack':int((tr['y'][p]==1).sum()),
                   'row_ids_hash':ids_hash(tr['row_id'][p])} for i,p in enumerate(parts)])
    write_json(output/'requests.json',{'invalid':invalid,'requests':{k:{'kind':v['kind'],'trigger':v['trigger'],'target':v['target'],
        'primary':v['primary'],'forget_count':len(v['forget'][1]),'forget_labels':np.bincount(v['forget'][1],minlength=2).tolist(),
        'target_retain_labels':np.bincount(v['target_retain'][1],minlength=2).tolist(),
        **{k2:v2 for k2,v2 in v['audit'].items() if k2 not in ('source_row_ids','nonmember_row_ids')},
        'source_row_ids_hash':ids_hash(v['audit']['source_row_ids']) if 'source_row_ids' in v['audit'] else None} for k,v in requests.items()}})
    clean=clean_cloud(ctx,mode,blocks_of(ctx.tr,parts),seeds,output,True); clients=list(range(len(parts)))
    cloud_rows,gate,runs,controls,dims_rows=[],{},[],[],[]
    for name,req in requests.items():
        cm,source=clouds_for(ctx,mode,name,req,seeds,clean,output,True); score=UtilityScorer(ctx,req,cm['retrain'])
        values={k:[score(m,i) for i,m in enumerate(ms)] for k,ms in cm.items()}
        cloud_rows.extend({'request':name,'cloud':k,'seed':s,**v} for k,vals in values.items() for s,v in zip(seeds,vals))
        gate[name]=R.gate_entry(values,req['target'],source)
        print(f'{name}: gate separable={gate[name]["separable"]} effect={gate[name]["effect_size"]:.2f}',flush=True)
        if not gate[name]['separable']: continue
        st=Request(ctx,req,cm)
        dims_rows.append({'request':name,'knn_seconds':st.knn_seconds,**{f'dead_dims_{k}':'/'.join(map(str,v)) for k,v in st.dims.items()}})
        for i,(s,orig) in enumerate(zip(seeds,cm['original'])):
            budgets={0}
            for arm,family,pre,post,info in st.seed_models(orig,s,i):
                steps=D.control_budget(info['backward_samples']) if family=='local' else 0; budgets.add(steps)
                common={'request':name,'target':req['target'],'seed':s,'arm':arm,'family':family,'control_steps':steps,**info}
                for stage,m in [('pre',pre),('post',post)]: runs.append({**common,'stage':stage,**score(m,i)})
            for kind,steps,pre,post,local,rec in st.controls(orig,s,budgets):
                for stage,m in [('pre',pre),('post',post)]:
                    controls.append({'request':name,'seed':s,'kind':kind,'control_steps':steps,'stage':stage,
                                     'local_seconds':local,'recovery_seconds':rec,**score(m,i)})
            print(f'{name}: seed {i+1}/{len(seeds)} ({time.perf_counter()-started:.0f}s)',flush=True)
    clouds=pd.DataFrame(cloud_rows); runs=pd.DataFrame(runs); controls=pd.DataFrame(controls)
    clouds.to_csv(output/'clouds.csv',index=False); runs.to_csv(output/'fu_runs.csv',index=False)
    controls.to_csv(output/'controls.csv',index=False); pd.DataFrame(dims_rows).to_csv(output/'request_state.csv',index=False)
    write_json(output/'gate.json',gate)
    admitted={k:v['target'] for k,v in gate.items() if v['separable']}
    decision,verdict=decide(clouds,runs,controls,admitted,clients,one_row_floor(mode)) if len(runs) else (pd.DataFrame(),{})
    decision.to_csv(output/'decision.csv',index=False); write_json(output/'verdict.json',verdict)
    check=crosscheck(runs,controls) if mode=='existing' and len(runs) else {'available':False}
    write_json(output/'status.json',{'completed':True,'mode':mode,'seeds':len(seeds),'requests':list(requests),
        'invalid_requests':list(invalid),'requests_passing_gate':list(admitted),'targets_passing_gate':sorted(set(admitted.values())),
        'arms_passing':[k for k,v in verdict.items() if v['pass']],
        'recommended_arm_pass':verdict.get(CONFIG['recommended_arm'],{}).get('pass'),'deployable_crosscheck':check,
        'total_seconds':time.perf_counter()-started,'selection_performed':False,'unlearning_success_claimed':False})
    finalize(output)
    return verdict

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--mode',choices=['existing','confirm','confirm2'],required=True)
    p.add_argument('--seeds',type=int,default=None,help='use only the first N seeds (smoke runs only)')
    p.add_argument('--requests',nargs='+',default=None,help='restrict to these requests (smoke runs only; recorded in the manifest)')
    a=p.parse_args(); run(a.output,a.mode,a.seeds,a.requests)
