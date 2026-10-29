"""Metadata-only requests, no model performance in selection."""
import numpy as np
from src.data.federated import forget_request


def build_request(train,parts,target,fraction,design,signatures,seed):
    if not 0 < fraction < 1: raise ValueError('Fraction must be strictly between zero and one')
    local=parts[target]
    if design=='random_stratified':
        f,r=forget_request(train['y'],local,fraction,seed)
    elif design=='low_overlap_groups':
        external=np.setdiff1d(np.arange(len(train['y'])),local)
        external_signatures=set(signatures[external])
        rng=np.random.default_rng(seed)
        unique=np.unique(train['group_id'][local]); rng.shuffle(unique)
        groups={g:[] for g in unique}
        for i in local: groups[train['group_id'][i]].append(i)
        ordered=sorted(unique,key=lambda g:sum(signatures[i] in external_signatures for i in groups[g])/len(groups[g]))
        capacity=int(np.floor(len(local)*fraction)); selected=[]
        for g in ordered:
            if len(selected)+len(groups[g])<=capacity: selected.extend(groups[g])
        f=np.array(sorted(selected),dtype=int); r=np.setdiff1d(local,f)
    else: raise ValueError('Unknown request design')
    if not len(f) or not len(r) or set(train['y'][f])!={0,1} or set(train['y'][r])!={0,1}:
        raise ValueError('Request has empty/single-class F or R; no resampling')
    global_r=np.setdiff1d(np.arange(len(train['y'])),f); rs=set(signatures[global_r])
    audit={'target':target,'design':design,'requested_fraction':fraction,'actual_fraction':len(f)/len(local),
           'forget_count':len(f),'retain_count':len(r),'forget_labels':np.bincount(train['y'][f],minlength=2).tolist(),
           'retain_labels':np.bincount(train['y'][r],minlength=2).tolist(),'forget_groups':len(set(train['group_id'][f])),
           'feature_equivalent_overlap_fraction':sum(s in rs for s in signatures[f])/len(f)}
    return f,r,audit
