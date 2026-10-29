"""Preregistered auxiliary budget correction and fresh held-out membership validation."""
import argparse
import copy
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
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from scripts.experiment_context import Context,CASES,finalize
from scripts.run_controlled_diagnostics import family_match
from src.eval.shadow_mia import auxiliary_buckets,features
from src.eval.membership_validation import score_summary,group_intervals,usable,roc_data,clean_auxiliary_roles,equivalence_components,logit_margin,stable_baselines
from src.eval.metrics import utility,membership_score
from src.fl.fedavg_core import model,state,fedavg,ClientWorker,predict,get_device
from src.data.prepare import write_json

def audit_model(m,d,a,b,label):
    rows=[]
    for role,idx in [('member',a),('nonmember',b)]:
        p=predict(m,d['X'][idx]); loss=-stable_baselines(d['y'][idx],logit_margin(m,d['X'][idx]))['negative_loss']
        rows.append({'model':label,'role':role,'count':len(idx),'normal':int((d['y'][idx]==0).sum()),
                     'attack':int((d['y'][idx]==1).sum()),'mean_logit_ce':float(loss.mean()),
                     **{k:v for k,v in utility(d['y'][idx],p).items() if k!='confusion_matrix'},
                     **{f'loss_q{q}':float(np.quantile(loss,q/100)) for q in [0,10,50,90,100]}})
    return rows

def run(output,shadow_cache=None):
    ctx=Context(); cfg=ctx.cfg; aux=ctx.data['calibration']; seed=cfg['seed']; started=time.perf_counter()
    device=get_device()
    print(f'Using device: {device}',flush=True)
    ctx.manifest_for(output,'membership_validation',{'cases':CASES,'shadow_rounds':15,'fixed_fprs':[.01,.05],
        'group_bootstrap_draws':200,'selection':'no attack selection or target-test tuning',
        'fresh_final_groups':'previously unused bucket7; seeded group split','shadow_cache':str(shadow_cache) if shadow_cache else None,
        'baseline_numerics':'stable logit-derived negative CE and absolute margin (monotonic max-confidence)'})
    if shadow_cache:
        for name,sha in json.loads((shadow_cache/'artifact_hashes.json').read_text()).items():
            from src.data.prepare import digest
            if digest(shadow_cache/name)!=sha: raise ValueError('Modified shadow cache')
    old_buckets=auxiliary_buckets(aux['group_id'],seed+600)
    buckets,components,role_audit=clean_auxiliary_roles(aux,old_buckets)
    np.savez_compressed(output/'excluded_auxiliary_ids.npz',row_ids=aux['row_id'][role_audit.pop('excluded_indices')])
    write_json(output/'auxiliary_role_audit.json',role_audit)
    aux={**aux,'group_id':components}
    train_components=equivalence_components(ctx.tr)
    test_components=equivalence_components(ctx.data['test'])
    # Explicit encoded-feature disjointness in addition to the frozen split audits.
    encoded=[set(row.tobytes() for row in aux['X'][idx]) for idx in buckets]
    for i in range(8):
        for j in range(i):
            if encoded[i]&encoded[j]: raise AssertionError('Auxiliary encoded-feature overlap')
    aux_encoded=set.union(*encoded)
    for s in ['train','validation','test']:
        if aux_encoded & set(row.tobytes() for row in ctx.data[s]['X']): raise AssertionError('Target/auxiliary overlap')
    groups=np.unique(aux['group_id'][buckets[7]]); np.random.default_rng(seed+701).shuffle(groups)
    a_groups=set(groups[:len(groups)//2]); final_a=np.array([i for i in buckets[7] if aux['group_id'][i] in a_groups])
    final_b=np.setdiff1d(buckets[7],final_a)
    pairs=[(buckets[0],buckets[1]),(buckets[2],buckets[3]),(buckets[4],buckets[5]),(final_a,final_b)]
    np.savez_compressed(output/'roles.npz',**{f'shadow{i}_{role}':aux['row_id'][idx]
        for i,pair in enumerate(pairs) for role,idx in zip(['members','nonmembers'],pair)},calibration=aux['row_id'][buckets[6]])
    np.savez_compressed(output/'role_components.npz',**{f'shadow{i}_{role}':aux['group_id'][idx]
        for i,pair in enumerate(pairs) for role,idx in zip(['members','nonmembers'],pair)},calibration=aux['group_id'][buckets[6]])
    if shadow_cache:
        with np.load(output/'roles.npz') as new, np.load(shadow_cache/'roles.npz') as old:
            for k in new.files: np.testing.assert_array_equal(new[k],old[k])
    old_source=Path(ctx.manifest['source']); original_logs=json.loads((old_source/'original_training.json').read_text())
    target_steps=sum(c['steps'] for r in original_logs for c in r['clients'])
    target_average_steps=target_steps/(len(original_logs)*5)
    fitx=[]; fity=[]; evaluation_pairs={}; coverage={}; audits=[]; budgets=[]
    original=ctx.load(CASES[0],'original')
    # Target train/validation are only audited; never used to fit or select attack settings.
    for s,role in [('train','member'),('validation','nonmember')]:
        d=ctx.data[s]; p=predict(original,d['X']); loss=-stable_baselines(d['y'],logit_margin(original,d['X']))['negative_loss']
        audits.append({'model':'target_original','role':role,'count':len(d['y']),'mean_logit_ce':float(loss.mean()),
                       'normal':int((d['y']==0).sum()),'attack':int((d['y']==1).sum()),
                       **{k:v for k,v in utility(d['y'],p).items() if k!='confusion_matrix'},
                       **{f'loss_q{q}':float(np.quantile(loss,q/100)) for q in [0,10,50,90,100]}})
    for i,(a,b) in enumerate(pairs):
        if set(aux['y'][a])!={0,1} or set(aux['y'][b])!={0,1}: raise ValueError('Degenerate shadow labels')
        if set(aux['group_id'][a])&set(aux['group_id'][b]): raise AssertionError('Shadow group leakage')
        if i<3:
            old=model(aux['X'].shape[1],cfg['hidden'],seed)
            old.load_state_dict(torch.load(ctx.source/'attack'/f'shadow_{i}.pt',weights_only=True)['state_dict'])
            audits.extend(audit_model(old,aux,old_buckets[2*i],old_buckets[2*i+1],f'old_shadow{i}'))
        rng=np.random.default_rng(seed+i); ci=[[] for _ in range(5)]
        for c in [0,1]:
            for dest,chunk in zip(ci,np.array_split(rng.permutation(a[aux['y'][a]==c]),5)): dest.extend(chunk)
        workers=[ClientWorker(j,aux['X'][idx],aux['y'][idx],aux['row_id'][idx]) for j,idx in enumerate(ci)]
        batch=max(8,int(np.floor(len(a)/5/target_average_steps)))
        fl={**cfg['fl'],'batch_size':batch}
        if shadow_cache:
            ck=torch.load(shadow_cache/f'shadow{i}.pt',weights_only=True)
            if ck['fl_config']!=fl: raise ValueError('Shadow configuration changed')
            m=model(aux['X'].shape[1],cfg['hidden'],seed+700+i); m.load_state_dict(ck['state_dict'])
            logs=json.loads((shadow_cache/f'shadow{i}_training.json').read_text())
        else:
            m,logs=fedavg(model(aux['X'].shape[1],cfg['hidden'],seed+700+i),workers,fl,15,seed+700+i)
        torch.save({'state_dict':state(m),'fl_config':fl},output/f'shadow{i}.pt'); write_json(output/f'shadow{i}_training.json',logs)
        budgets.append({'shadow':i,'member_count':len(a),'nonmember_count':len(b),'batch_size':batch,'cached':shadow_cache is not None,
                        'optimizer_steps':sum(c['steps'] for r in logs for c in r['clients']),
                        'target_optimizer_steps':target_steps,'rounds':15,'local_epochs':fl['local_epochs'],
                        'client_counts':[len(idx) for idx in ci],'wall_seconds':sum(r['wall_seconds'] for r in logs)})
        audits.extend(audit_model(m,aux,a,b,f'improved_shadow{i}'))
        mi,ni,cov=family_match(a,aux,{'y':aux['y'][b]},
                             {'train':ctx.types['calibration'],'test':ctx.types['calibration'][b]},seed)
        ni=b[ni]; coverage[f'shadow{i}']=cov
        np.savez_compressed(output/f'shadow{i}_matched_ids.npz',member=aux['row_id'][mi],nonmember=aux['row_id'][ni])
        mp,npred=predict(m,aux['X'][mi]),predict(m,aux['X'][ni])
        if i<2:
            fitx.append(np.r_[features(aux['y'][mi],mp),features(aux['y'][ni],npred)])
            fity.append(np.r_[np.ones(len(mi)),np.zeros(len(ni))])
        else: evaluation_pairs['development' if i==2 else 'final']=(mi,ni,mp,npred,logit_margin(m,aux['X'][mi]),logit_margin(m,aux['X'][ni]))
        print(f'Shadow {i}: batch {batch}, steps {budgets[-1]["optimizer_steps"]}',flush=True)
    classifier=make_pipeline(StandardScaler(),LogisticRegression(C=1.,max_iter=1000,random_state=seed))
    classifier.fit(np.concatenate(fitx),np.concatenate(fity))
    if classifier[-1].n_iter_.max()>=1000: raise RuntimeError('Nonconverged attack')
    def scores(y,p,margin):
        return {'learned':classifier.predict_proba(features(y,p))[:,1],
                **stable_baselines(y,margin)}
    calibration_idx=buckets[6]; cp=predict(original,aux['X'][calibration_idx])
    thresholds={name:{str(fpr):float(np.quantile(v,1-fpr,method='higher')) for fpr in [.01,.05]}
                for name,v in scores(aux['y'][calibration_idx],cp,logit_margin(original,aux['X'][calibration_idx])).items()}
    write_json(output/'attack.json',{'scaler_mean':classifier[0].mean_.tolist(),'scaler_scale':classifier[0].scale_.tolist(),
        'coef':classifier[-1].coef_.tolist(),'intercept':classifier[-1].intercept_.tolist(),'thresholds':thresholds,
        'score_direction':'higher means member; never flipped','features':['log_p_true','entropy','margin','label'],
        'threshold_rule':'strict > original bucket6 quantile; same threshold for all models',
        'calibration_realized_fpr':{name:{f:float(np.mean(v>t)) for f,t in thresholds[name].items()}
             for name,v in scores(aux['y'][calibration_idx],cp,logit_margin(original,aux['X'][calibration_idx])).items()}})
    shadow_results=[]; rocs={}; final_results={}
    for stage,(mi,ni,mp,npred,mm,nm) in evaluation_pairs.items():
        sm,sn=scores(aux['y'][mi],mp,mm),scores(aux['y'][ni],npred,nm)
        for name in sm:
            result={'stage':stage,'attack':name,**score_summary(sm[name],sn[name],thresholds[name]),
                    **group_intervals(sm[name],sn[name],aux['group_id'][mi],aux['group_id'][ni],thresholds[name])}
            shadow_results.append(result); rocs[f'shadow/{stage}/{name}']=roc_data(sm[name],sn[name])
            if stage=='final': final_results[name]=result
        np.savez_compressed(output/f'{stage}_scores.npz',**{f'{name}_{role}':v for role,ss in [('member',sm),('nonmember',sn)] for name,v in ss.items()})
    pd.DataFrame(audits).to_csv(output/'model_audit.csv',index=False)
    write_json(output/'shadow_budgets.json',budgets); write_json(output/'coverage.json',coverage)
    pd.DataFrame(shadow_results).to_csv(output/'shadow_validation.csv',index=False)
    # Constructed positive control on development auxiliaries only. Fixed budget, no retries.
    def unique_examples(idx):
        _,first=np.unique(aux['X'][idx],axis=0,return_index=True)
        return idx[np.sort(first)[:64]]
    pm,pn=unique_examples(buckets[4]),unique_examples(buckets[5])
    rng=np.random.default_rng(seed+999); py=rng.permutation(np.tile([0,1],32)); ny=rng.permutation(np.tile([0,1],32))
    positive=model(aux['X'].shape[1],cfg['hidden'],seed+999)
    opt=torch.optim.Adam(positive.parameters(),lr=.01)
    x=torch.as_tensor(aux['X'][pm],dtype=torch.float32,device=device)
    y=torch.as_tensor(py,device=device)
    if shadow_cache:
        positive.load_state_dict(torch.load(shadow_cache/'positive_control.pt',weights_only=True)['state_dict'])
        with np.load(shadow_cache/'positive_control_ids_labels.npz') as z:
            np.testing.assert_array_equal(z['member_ids'],aux['row_id'][pm]); np.testing.assert_array_equal(z['nonmember_ids'],aux['row_id'][pn])
            np.testing.assert_array_equal(z['member_labels'],py); np.testing.assert_array_equal(z['nonmember_labels'],ny)
    else:
        for step in range(2000):
            opt.zero_grad(); loss=torch.nn.functional.cross_entropy(positive(x),y); loss.backward(); opt.step()
    sm,sn=scores(py,predict(positive,aux['X'][pm]),logit_margin(positive,aux['X'][pm])),scores(ny,predict(positive,aux['X'][pn]),logit_margin(positive,aux['X'][pn]))
    write_json(output/'positive_control_diagnostics.json',{
        'member_accuracy':float(np.mean((predict(positive,aux['X'][pm])>=.5)==py)),
        'nonmember_accuracy':float(np.mean((predict(positive,aux['X'][pn])>=.5)==ny)),
        'member_mean_ce':float(-sm['negative_loss'].mean()),'nonmember_mean_ce':float(-sn['negative_loss'].mean()),
        'optimizer_steps':2000,'cached':shadow_cache is not None,'no_retry_or_control_selection':True})
    controls=[]
    for name in sm:
        controls.append({'control':'random_label_memorization','attack':name,**score_summary(sm[name],sn[name],thresholds[name])})
        controls.append({'control':'identical_score_distribution','attack':name,**score_summary(sn[name],sn[name],thresholds[name])})
    torch.save({'state_dict':state(positive),'steps':2000},output/'positive_control.pt')
    np.savez_compressed(output/'positive_control_ids_labels.npz',member_ids=aux['row_id'][pm],nonmember_ids=aux['row_id'][pn],member_labels=py,nonmember_labels=ny)
    pd.DataFrame(controls).to_csv(output/'controls.csv',index=False)
    results=[]; all_coverage={}
    for case in CASES:
        chosen=ctx.selections[case]['chosen']
        names={'original':'original','retain_only':'no_update_post','gd_recovery':'gd_only_post',
               'alternating_recovery':chosen+'_post','fu_kd':'ga_gd_with_kd_post','retrain':'retrain'}
        d=ctx.data['test']; tindex={v:i for i,v in enumerate(d['row_id'])}
        with np.load(ctx.source/case/'mia_ids.npz') as z:
            cohorts={c:(np.array([ctx.index[v] for v in z[c+'_member']]),np.array([tindex[v] for v in z[c+'_nonmember']]))
                     for c in ['F','R_all','R_group']}
        all_coverage[case]=json.loads((ctx.source/case/'mia_coverage.json').read_text())
        # Strengthen the retained control if old group IDs did not close encoded equivalence.
        f,retained,_=ctx.request(case); ri=np.concatenate(retained); fc=set(train_components[f])
        independent=ri[np.array([g not in fc for g in train_components[ri]])]
        mi,ni,cov=family_match(independent,ctx.tr,d,{'train':ctx.types['train'],'test':ctx.types['test']},seed)
        old_m=cohorts['R_group'][0]
        all_coverage[case]['old_R_group_rows_sharing_F_component']=int(sum(g in fc for g in train_components[old_m]))
        all_coverage[case]['R_group_component_matched']=cov
        cohorts['R_group']=(mi,ni)
        predictions={name:(predict(ctx.load(case,ck),ctx.tr['X']),predict(ctx.load(case,ck),d['X'])) for name,ck in names.items()}
        margins={name:(logit_margin(ctx.load(case,ck),ctx.tr['X']),logit_margin(ctx.load(case,ck),d['X'])) for name,ck in names.items()}
        for cohort,(mi,ni) in cohorts.items():
            rm,rn=scores(ctx.tr['y'][mi],predictions['retrain'][0][mi],margins['retrain'][0][mi]),scores(d['y'][ni],predictions['retrain'][1][ni],margins['retrain'][1][ni])
            for name,(pt,pn) in predictions.items():
                sm,sn=scores(ctx.tr['y'][mi],pt[mi],margins[name][0][mi]),scores(d['y'][ni],pn[ni],margins[name][1][ni])
                np.savez_compressed(output/f'{case}_{cohort}_{name}_scores.npz',member_ids=ctx.tr['row_id'][mi],nonmember_ids=d['row_id'][ni],
                        **{f'{atk}_{role}':v for role,ss in [('member',sm),('nonmember',sn)] for atk,v in ss.items()})
                for attack in sm:
                    ref=score_summary(rm[attack],rn[attack],thresholds[attack])
                    result={'case':case,'cohort':cohort,'model':name,'attack':attack,
                        **score_summary(sm[attack],sn[attack],thresholds[attack]),
                        **group_intervals(sm[attack],sn[attack],train_components[mi],test_components[ni],thresholds[attack],reference=(rm[attack],rn[attack])),
                        'low_fpr_sample_warning':len(ni)<1000}
                    for metric in ['auc','tpr_0.01','tpr_0.05','actual_fpr_0.01','actual_fpr_0.05']:
                        result[metric+'_minus_retrain']=result[metric]-ref[metric]
                    results.append(result); rocs[f'{case}/{cohort}/{name}/{attack}']=roc_data(sm[attack],sn[attack])
        print(f'Completed MIA {case}',flush=True)
    for result in results:
        control=next(r for r in results if r['case']==result['case'] and r['attack']==result['attack'] and r['model']=='original' and r['cohort']=='R_group')
        result['interpretation']='EMPIRICAL_ATTACK_DETECTABLE_NOT_DELETION_PROOF' if usable(final_results[result['attack']]) and usable(control) else 'MIA_INCONCLUSIVE'
    pd.DataFrame(results).to_csv(output/'comparison.csv',index=False)
    write_json(output/'target_coverage.json',all_coverage); write_json(output/'roc.json',rocs)
    fig,axes=plt.subplots(1,3,figsize=(14,4))
    for ax,stage in zip(axes,['shadow/final',CASES[0]+'/F/original',CASES[1]+'/F/original']):
        for attack in thresholds:
            curve=rocs[stage+'/'+attack]; ax.plot(curve['fpr'],curve['tpr'],label=attack)
        ax.plot([0,1],[0,1],'k--',alpha=.4); ax.set(xlabel='FPR',ylabel='TPR',title=stage); ax.legend(fontsize=8)
    fig.tight_layout(); fig.savefig(output/'roc_controls.png',dpi=160); plt.close(fig)
    for case in CASES:
        fig,axes=plt.subplots(1,3,figsize=(14,4))
        for ax,attack in zip(axes,thresholds):
            for name in names:
                curve=rocs[f'{case}/F/{name}/{attack}']; ax.plot(curve['fpr'],curve['tpr'],label=name)
            ax.plot([0,1],[0,1],'k--',alpha=.4); ax.set(xlabel='FPR',ylabel='TPR',title=attack); ax.legend(fontsize=7)
        fig.suptitle(case+'; fixed forget cohort'); fig.tight_layout(); fig.savefig(output/f'{case}_roc.png',dpi=160); plt.close(fig)
    write_json(output/'status.json',{'completed':True,'target_rows':len(results),'shadow_models':4,
        'encoded_auxiliary_leakage':False,'test_used_for_selection':False,'total_seconds':time.perf_counter()-started,
        'unlearning_success_claimed':False,'target_seed_count':1,'new_shadow_fits':0 if shadow_cache else 4})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shadow-cache',type=Path)
    a=p.parse_args(); run(a.output,a.shadow_cache)
