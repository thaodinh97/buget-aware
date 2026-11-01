"""Matched-canary membership inference on the deployable-forget models.

Members are the 500 canaries of each canary request; nonmembers are 500 other attack rows of the
same client, stamped and relabelled the same way but never trained on. Both are uniform draws from
that client's attack rows, so for a model that never saw the canaries (retrain) the AUC is 0.5 up to
sampling. The original model is the positive control: the attack has power on a request only if the
original's AUC exceeds retrain's, paired over seeds with the 95% CI above 0. Unlearned models are
rebuilt with the deployable-forget code and checked bit-for-bit against its stored metrics.
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
from scripts import run_deployable_forget as D, run_forget_signal as R
from src.data.canary import matched_nonmembers
from src.data.prepare import digest,write_json
from src.eval.fu_decision import mean_ci
from src.eval.membership_validation import logit_margin,score_summary
from src.fl.fedavg_core import fedavg
from src.fu.bounded_forget import forget_ce,forget_prob
from src.fu.neighbor_reference import knn_indices

DEPLOYABLE=Path('runs/deployable_forget_seed42')
CONFIG={'nonmember_count':500,'nonmember_seed_offset':17,'fpr':.05,
        'score':'log p(normal), stable from the logit margin; higher = more member-like',
        'arms':['ga_proj@joint_aux_prob','nkd_proj@joint_aux_prob','ga_adam@joint_aux_prob']}

def check_deployable():
    for name,sha in json.loads((DEPLOYABLE/'artifact_hashes.json').read_text()).items():
        if digest(DEPLOYABLE/name)!=sha: raise ValueError(f'Changed deployable-forget artifact {name}')

def canary_sets(ctx,req):
    """Members (the canaries) and matched nonmembers for one canary request."""
    c=req['target']; src=np.array([ctx.index[r] for r in req['audit']['source_row_ids']])
    nx,_,picked=matched_nonmembers(ctx.tr['X'],ctx.tr['y'],ctx.parts[c],src,CONFIG['nonmember_count'],
                                   np.array(req['audit']['trigger_dims']),ctx.cfg['seed']+CONFIG['nonmember_seed_offset']+c)
    return req['forget'][0],nx,picked

def attack(m,members,nonmembers):
    sm,sn=-np.logaddexp(0,logit_margin(m,members)),-np.logaddexp(0,logit_margin(m,nonmembers))
    return {'auc':score_summary(sm,sn,{})['auc'],'tpr_at_fpr':float((sm>np.quantile(sn,1-CONFIG['fpr'])).mean()),
            'member_loss':float(-sm.mean()),'nonmember_loss':float(-sn.mean())}

def rebuild(ctx,name,req,seeds,stored,controls):
    """Yield (seed, {model name: model}) with every unlearned and retain-only model checked against the stored run."""
    cm,_=R.cloud_models(ctx,name,req,seeds,D.FORGET_SIGNAL,False); score=R.Scorer(ctx,req,cm['retrain'])
    fw,rw,_,recovery=R.workers_for(req); pools=D.pools_for(ctx,req); fx,fy=req['forget'][:2]
    nb={k:knn_indices(fx,p,D.CONFIG['knn_k']) for k,p in pools.items()}
    rce,rp=(np.array([f(m,fw) for m in cm['retrain']]) for f in (forget_ce,forget_prob))
    for i,(s,orig) in enumerate(zip(seeds,cm['original'])):
        ref,ref_rows,_=D.reference_model(ctx,req,s)
        _,levels,targets=D.references(orig,fx,fy,pools,nb,rce,rp,i,fw.device,ref)
        models={'original':orig,'retrain':cm['retrain'][i]}
        models['retain_only'],_=fedavg(orig,recovery,ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s+900000)
        expected=controls[(controls.request==name)&(controls.seed==s)&(controls.control_steps==0)&(controls.stage=='post')].primary.iloc[0]
        if abs(score(models['retain_only'],i)['primary']-expected)>1e-9: raise AssertionError(f'{name}/{s}: retain_only differs')
        for arm in CONFIG['arms']:
            op,stop=arm.split('@')
            models[arm]=D.arm_runs(orig,fw,rw,recovery,ctx.cfg['fl'],ctx.cfg['recovery_rounds'],s,op,stop,levels,targets,
                                   (ref_rows,0.) if D.charged(stop) else (0,0.))[0][2]
            expected=stored[(stored.request==name)&(stored.seed==s)&(stored.arm==arm)&(stored.stage=='post')].primary.iloc[0]
            if abs(score(models[arm],i)['primary']-expected)>1e-9: raise AssertionError(f'{name}/{s}: {arm} differs from the stored run')
        yield s,models

def summarize(rows):
    out=[]
    for name,g in rows.groupby('request',sort=False):
        base=g[g.model=='retrain'].sort_values('seed').auc.to_numpy()
        for model,h in g.groupby('model',sort=False):
            auc=h.sort_values('seed').auc.to_numpy(); m,lo,hi=mean_ci(auc); d,dlo,dhi=mean_ci(auc-base)
            out.append({'request':name,'model':model,'seeds':len(auc),'auc_mean':m,'auc_ci_low':lo,'auc_ci_high':hi,
                        'auc_minus_retrain_mean':d,'auc_minus_retrain_ci_low':dlo,'auc_minus_retrain_ci_high':dhi,
                        'tpr_at_fpr_mean':h.tpr_at_fpr.mean(),'member_loss_mean':h.member_loss.mean(),'nonmember_loss_mean':h.nonmember_loss.mean()})
    summary=pd.DataFrame(out)
    power={q:bool(s[s.model=='original'].auc_minus_retrain_ci_low.iloc[0]>0) for q,s in summary.groupby('request',sort=False)}
    summary['attack_has_power']=summary.request.map(power)
    summary['verdict']=np.where(~summary.attack_has_power,'MIA_INCONCLUSIVE',
        np.where((summary.auc_minus_retrain_ci_low<=0)&(summary.auc_minus_retrain_ci_high>=0),'RETRAIN_LIKE','DIFFERS_FROM_RETRAIN'))
    return summary,power

def run(output):
    use_run_device(DEPLOYABLE); check_deployable(); D.check_source(); started=time.perf_counter()
    ctx=Context(); seeds=json.loads((DEPLOYABLE/'manifest.json').read_text())['protocol']['seeds']
    ctx.manifest_for(output,'canary_mia',{**CONFIG,'seeds':seeds,'deployable_source':str(DEPLOYABLE),
        'deployable_artifact_hashes_sha256':digest(DEPLOYABLE/'artifact_hashes.json'),
        'power_rule':'attack has power on a request iff the paired 95% CI of AUC(original) - AUC(retrain) over seeds lies above 0',
        'models':'original, retrain (canary-free seed-floor original), retain_only (0 local steps + standard recovery), arms (standard recovery)',
        'tpr_at_fpr':'descriptive ROC point at the nonmember score quantile; not a calibrated operating threshold'})
    stored=pd.read_csv(DEPLOYABLE/'fu_runs.csv'); controls=pd.read_csv(DEPLOYABLE/'controls.csv')
    requests={k:v for k,v in R.build_requests(ctx).items() if v['kind']=='canary'}; rows=[]; audit={}
    for name,req in requests.items():
        members,nonmembers,picked=canary_sets(ctx,req)
        audit[name]={'target':req['target'],'members':len(members),'nonmember_row_ids':ctx.tr['row_id'][picked].tolist()}
        for s,models in rebuild(ctx,name,req,seeds,stored,controls):
            rows.extend({'request':name,'target':req['target'],'seed':s,'model':k,**attack(m,members,nonmembers)} for k,m in models.items())
        print(f'{name}: done ({time.perf_counter()-started:.0f}s)',flush=True)
    rows=pd.DataFrame(rows); rows.to_csv(output/'mia_rows.csv',index=False); write_json(output/'cohorts.json',audit)
    summary,power=summarize(rows); summary.to_csv(output/'summary.csv',index=False)
    write_json(output/'status.json',{'completed':True,'seeds':len(seeds),'requests':list(requests),'attack_has_power':power,
        'rebuilt_models_match_stored_run':True,'total_seconds':time.perf_counter()-started,'unlearning_success_claimed':False})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
