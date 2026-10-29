"""Fixed-score MIA and cluster uncertainty; no target-based score selection."""
import numpy as np
import torch
from sklearn.metrics import roc_auc_score,roc_curve
from src.data.prepare import connected_groups

def logit_margin(model,x,batch_size=4096):
    model.eval(); values=[]; device=next(model.parameters()).device
    with torch.no_grad():
        for i in range(0,len(x),batch_size):
            logits=model(torch.as_tensor(x[i:i+batch_size],dtype=torch.float32,device=device)).double()
            values.append((logits[:,1]-logits[:,0]).cpu().numpy())
    return np.concatenate(values)

def stable_baselines(y,margin):
    return {'negative_loss':-np.logaddexp(0,-(2*np.asarray(y)-1)*margin),
            'confidence':np.abs(margin)} # strictly monotone in max softmax confidence

def equivalence_components(data):
    return connected_groups(data['group_id'],[row.tobytes() for row in data['X']])

def clean_auxiliary_roles(data,buckets):
    """Preserve prior roles; exclude whole components crossing roles, never move rows."""
    components=equivalence_components(data)
    roles=np.full(len(components),-1,dtype=int)
    for i,idx in enumerate(buckets): roles[idx]=i
    if (roles<0).any(): raise ValueError('Incomplete auxiliary role assignment')
    membership={}
    for component,role in zip(components,roles): membership.setdefault(component,set()).add(int(role))
    bad={k for k,v in membership.items() if len(v)>1}
    keep=np.array([c not in bad for c in components])
    cleaned=[idx[keep[idx]] for idx in buckets]
    return cleaned,components,{'excluded_rows':int((~keep).sum()),'cross_role_components':len(bad),
        'rows_before':[len(idx) for idx in buckets],'rows_after':[len(idx) for idx in cleaned],
        'excluded_indices':np.flatnonzero(~keep).tolist()}

def score_summary(member,nonmember,thresholds):
    m,n=np.asarray(member),np.asarray(nonmember)
    if not len(m) or not len(n): raise ValueError('Empty attack cohort')
    result={'auc':float(roc_auc_score(np.r_[np.ones(len(m)),np.zeros(len(n))],np.r_[m,n])),
            'member_count':len(m),'nonmember_count':len(n)}
    for fpr,t in thresholds.items():
        tp=float((m>t).mean()); fp=float((n>t).mean())
        result.update({f'tpr_{fpr}':tp,f'actual_fpr_{fpr}':fp,f'balanced_accuracy_{fpr}':(tp+1-fp)/2,
                       f'threshold_{fpr}':float(t)})
    return result

def group_intervals(m,n,mg,ng,thresholds,reference=None,seed=123,draws=200):
    # Same seed/groups yields paired resampling for every comparator.
    _,mi=np.unique(mg,return_inverse=True); _,ni=np.unique(ng,return_inverse=True)
    km,kn=mi.max()+1,ni.max()+1
    result={'member_groups':int(km),'nonmember_groups':int(kn),'bootstrap_draws':draws,
            'ci_scope':'conditional fixed attack/threshold; group clusters; no training-seed uncertainty'}
    if min(km,kn)<20:
        return {**result,'ci_status':'INSUFFICIENT_GROUP_SUPPORT'}
    rng=np.random.default_rng(seed); samples={k:[] for k in ['auc','auc_minus_retrain']}
    for fpr in thresholds: samples.update({f'tpr_{fpr}':[],f'actual_fpr_{fpr}':[]})
    truth=np.r_[np.ones(len(m)),np.zeros(len(n))]; scores=np.r_[m,n]
    for _ in range(draws):
        wm=rng.multinomial(km,np.ones(km)/km)[mi]; wn=rng.multinomial(kn,np.ones(kn)/kn)[ni]
        weights=np.r_[wm,wn]
        auc=roc_auc_score(truth,scores,sample_weight=weights); samples['auc'].append(auc)
        if reference is not None:
            samples['auc_minus_retrain'].append(auc-roc_auc_score(truth,np.r_[reference[0],reference[1]],sample_weight=weights))
        for fpr,t in thresholds.items():
            samples[f'tpr_{fpr}'].append(np.average(m>t,weights=wm))
            samples[f'actual_fpr_{fpr}'].append(np.average(n>t,weights=wn))
    for name,values in samples.items():
        if values:
            result[name+'_ci_low'],result[name+'_ci_high']=map(float,np.quantile(values,[.025,.975]))
    return {**result,'ci_status':'GROUP_BOOTSTRAP_95_PERCENT'}

def usable(result):
    return result['auc']>=.55 and result.get('auc_ci_low',0)>.5

def roc_data(m,n):
    fpr,tpr,_=roc_curve(np.r_[np.ones(len(m)),np.zeros(len(n))],np.r_[m,n])
    return {'fpr':fpr.tolist(),'tpr':tpr.tolist()}
