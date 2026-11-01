"""Independent auxiliary-group FL shadows and a fixed learned attack."""
import numpy as np
import torch
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from src.fl.fedavg_core import model,ClientWorker,fedavg,predict,state,get_device
from src.data.prepare import write_json
from src.profiling import CostProfile
from src.eval.membership_validation import family_match


def auxiliary_buckets(groups,seed):
    unique=np.unique(groups); np.random.default_rng(seed).shuffle(unique)
    mapping={g:j for j,chunk in enumerate(np.array_split(unique,8)) for g in chunk}
    return [np.flatnonzero(np.array([mapping[g] for g in groups])==j) for j in range(8)]


def features(y,p):
    p=np.clip(np.asarray(p,dtype=float),1e-7,1-1e-7)
    return np.column_stack([np.log(np.where(y==1,p,1-p)),-p*np.log(p)-(1-p)*np.log(1-p),abs(2*p-1),y])


def assess_scores(member,nonmember,threshold):
    if not len(member) or not len(nonmember): raise ValueError('Empty MIA cohort')
    return {'auc':float(roc_auc_score(np.r_[np.ones(len(member)),np.zeros(len(nonmember))],np.r_[member,nonmember])),
            'tpr_at_calibrated_fpr_0_01':float(np.mean(member>threshold)),
            'realized_nonmember_fpr':float(np.mean(nonmember>threshold)),
            'members':len(member),'nonmembers':len(nonmember),'threshold':threshold,
            'low_fpr_sample_warning':len(nonmember)<1000}


class LearnedAttack:
    def __init__(self,estimator,threshold,validation):
        self.estimator=estimator; self.threshold=threshold; self.validation=validation

    def scores(self,y,p): return self.estimator.predict_proba(features(y,p))[:,1]

    def assess(self,my,mp,ny,npred):
        return assess_scores(self.scores(my,mp),self.scores(ny,npred),self.threshold)


def train_attack(aux,aux_types,original,base_cfg,phase_cfg,output):
    output.mkdir()
    device=get_device().cuda if torch.cuda.is_available() else get_device()
    buckets=auxiliary_buckets(aux['group_id'],phase_cfg['seed']+600)
    for i,a in enumerate(buckets):
        if set(aux['y'][a])!={0,1}: raise ValueError(f'Degenerate auxiliary bucket {i}')
        for b in buckets[i+1:]:
            if set(aux['group_id'][a])&set(aux['group_id'][b]): raise AssertionError('Auxiliary overlap')
    np.savez_compressed(output/'auxiliary_roles.npz',**{f'bucket_{i}_{key}':aux[key][idx]
                         for i,idx in enumerate(buckets) for key in ['row_id','group_id']})
    fit_x=[]; fit_y=[]; hold=None; coverage={}; profiles={}
    pairs=phase_cfg['shadow_train_pairs']+[phase_cfg['shadow_validation_pair']]
    for shadow_id,(inside,outside) in enumerate(pairs):
        a,b=buckets[inside],buckets[outside]
        rng=np.random.default_rng(base_cfg['seed']+shadow_id)
        client_indices=[[] for _ in range(5)]
        for c in [0,1]:
            for dest,chunk in zip(client_indices,np.array_split(rng.permutation(a[aux['y'][a]==c]),5)): dest.extend(chunk)
        workers=[ClientWorker(i,aux['X'][idx],aux['y'][idx],aux['row_id'][idx],device=device)
                 for i,idx in enumerate(client_indices)]
        initial=model(aux['X'].shape[1],base_cfg['hidden'],base_cfg['seed']+700+shadow_id,device=device)
        with CostProfile() as prof:
            shadow,logs=fedavg(initial,workers,base_cfg['fl'],base_cfg['fl']['rounds'],base_cfg['seed']+700+shadow_id)
        profiles[str(shadow_id)]=prof.result(steps=sum(c['steps'] for e in logs for c in e['clients']),training_rounds=len(logs),
              exposures=len(a)*base_cfg['fl']['local_epochs']*len(logs),
              up=sum(e['payload_bytes_up'] for e in logs),down=sum(e['payload_bytes_down'] for e in logs))
        torch.save({'state_dict':state(shadow),'in_bucket':inside,'out_bucket':outside},output/f'shadow_{shadow_id}.pt')
        write_json(output/f'shadow_{shadow_id}_training.json',logs)
        member={'y':aux['y'][a]}; nonmember={'y':aux['y'][b]}
        mi,ni,cov=family_match(np.arange(len(a)),member,nonmember,{'train':aux_types[a],'test':aux_types[b]},base_cfg['seed'])
        coverage[str(shadow_id)]=cov
        x=np.concatenate([features(aux['y'][a[mi]],predict(shadow,aux['X'][a[mi]])),
                          features(aux['y'][b[ni]],predict(shadow,aux['X'][b[ni]]))])
        y=np.r_[np.ones(len(mi)),np.zeros(len(ni))]
        np.savez_compressed(output/f'shadow_{shadow_id}_attack_ids.npz',members=aux['row_id'][a[mi]],nonmembers=aux['row_id'][b[ni]])
        if shadow_id<len(phase_cfg['shadow_train_pairs']): fit_x.append(x); fit_y.append(y)
        else: hold=(x,y)
    classifier=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=base_cfg['seed']))
    classifier.fit(np.concatenate(fit_x),np.concatenate(fit_y))
    if classifier[-1].n_iter_.max()>=1000: raise RuntimeError('Attack failed to converge')
    threshold_idx=buckets[phase_cfg['attack_threshold_bucket']]
    scores=classifier.predict_proba(features(aux['y'][threshold_idx],predict(original,aux['X'][threshold_idx])))[:,1]
    threshold=float(np.quantile(scores,1-phase_cfg['nominal_mia_fpr'],method='higher'))
    hx,hy=hold; hs=classifier.predict_proba(hx)[:,1]
    validation=assess_scores(hs[hy==1],hs[hy==0],threshold)
    validation.update({'calibration_nonmembers':len(scores),'calibration_realized_fpr':float(np.mean(scores>threshold)),
                       'status':'weak_shadow_control' if validation['auc']<.55 else 'shadow_control_detectable_not_target_validation'})
    write_json(output/'attack.json',{'feature_names':['log_p_true','entropy','margin','label'],
               'scaler_mean':classifier[0].mean_.tolist(),'scaler_scale':classifier[0].scale_.tolist(),
               'coef':classifier[-1].coef_.tolist(),'intercept':classifier[-1].intercept_.tolist(),
               'classes':classifier[-1].classes_.tolist(),'threshold':threshold,'nominal_fpr':phase_cfg['nominal_mia_fpr'],
               'shadow_validation':validation,'training_pairs':phase_cfg['shadow_train_pairs'],
               'heldout_pair':phase_cfg['shadow_validation_pair'],'target_threshold_bucket':phase_cfg['attack_threshold_bucket'],
               'threshold_rule':'strict > original nonmember calibration quantile; frozen for all targets'})
    write_json(output/'coverage.json',coverage); write_json(output/'profiles.json',profiles)
    return LearnedAttack(classifier,threshold,validation)
