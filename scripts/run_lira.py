"""Paper-faithful LiRA (Carlini et al. 2022) on the fixed FU cohorts with balanced FedAvg shadows."""
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
from scripts.experiment_context import Context,CASES,finalize
from src.data.prepare import connected_groups,digest,write_json
from src.eval.lira import LiRA,lira_signal,balanced_masks,evaluate_mia
from src.eval.membership_validation import score_summary,group_intervals,usable,roc_data,equivalence_components
from src.fl.fedavg_core import model,state,fedavg,ClientWorker,get_device

VARIANTS={'online_global':('online','global'),'online_per_example':('online','per_example'),'offline_global':('offline','global')}
PRIMARY='online_global'; FPRS=[.001,.01,.05]; MAX_UNIT=8; LOO=8
COHORT_SOURCE=Path('runs/membership_validation_v3_seed42')

def units(x):
    # Rows with identical encoded features are one membership unit: they enter or leave a shadow together.
    return connected_groups([row.tobytes() for row in x])

def thresholds_from(nonmember):
    return {str(f):float(np.quantile(nonmember,1-f,method='higher')) for f in FPRS}

def grid_roc(m,n):
    # Large pooled control curves are stored on a fixed log-FPR grid to keep roc.json small.
    curve=roc_data(m,n); grid=np.unique(np.r_[0.,np.logspace(-5,0,400)])
    return {'fpr':grid.tolist(),'tpr':np.interp(grid,curve['fpr'],curve['tpr']).tolist()}

def summarize(sm,sn,mg,ng,thr,reference=None):
    roc=evaluate_mia(sm,sn,FPRS)
    return {**score_summary(sm,sn,thr),**group_intervals(sm,sn,mg,ng,thr,reference=reference),
            **{k:v for k,v in roc.items() if k.startswith('roc_')},
            **{f'low_fpr_sample_warning_{f}':len(sn)<10/f for f in FPRS}}

def run(output,n_shadows):
    ctx=Context(); cfg=ctx.cfg; seed=cfg['seed']; tr,te=ctx.tr,ctx.data['test']; started=time.perf_counter()
    print(f'Using device: {get_device()}',flush=True)
    for name,sha in json.loads((COHORT_SOURCE/'artifact_hashes.json').read_text()).items():
        if name.endswith('_original_scores.npz') and digest(COHORT_SOURCE/name)!=sha: raise ValueError(f'Changed cohort source: {name}')
    ctx.manifest_for(output,'lira',{'paper':'Carlini et al. 2022, Membership Inference Attacks From First Principles',
        'cases':CASES,'cohorts':['F','R_group'],'cohort_source':str(COHORT_SOURCE),'shadows_per_case':n_shadows,
        'membership':'balanced: every unit IN for exactly half of the shadows (Algorithm 1, Sec. IV-C)',
        'unit':'encoded-feature duplicate class; units larger than %d rows or (for F) touching non-forget rows are ineligible'%MAX_UNIT,
        'shadow_training':'same model/FedAvg config/rounds/client partition as target; seeds seed+5000+100*case+k',
        'signal':'z_y - logsumexp(z_other) in float64 (Sec. VI-A)','primary_variant':PRIMARY,'secondary_variants':[v for v in VARIANTS if v!=PRIMARY],
        'thresholds':'per case/variant from leave-one-shadow-out OUT scores; identical for all target models',
        'fprs':FPRS,'loo_shadows':LOO,'selection':'no variant, threshold or cohort chosen from target results'})
    tindex={v:i for i,v in enumerate(te['row_id'])}
    enc_tr,enc_te=units(tr['X']),units(te['X'])
    _,inverse,counts=np.unique(enc_tr,return_inverse=True,return_counts=True); unit_size=counts[inverse]
    clusters_tr=np.array([f'tr{c}' for c in equivalence_components(tr)]); clusters_te=np.array([f'te{c}' for c in equivalence_components(te)])
    if set(row.tobytes() for row in te['X']) & set(row.tobytes() for row in tr['X']): raise AssertionError('Train/test encoded overlap')
    client_of=np.empty(len(tr['y']),dtype=int)
    for i,p in enumerate(ctx.parts): client_of[p]=i
    names_base={'original':'original','retain_only':'no_update_post','gd_recovery':'gd_only_post',
                'fu_kd':'ga_gd_with_kd_post','retrain':'retrain'}
    comparison=[]; loo_rows=[]; eligibility={}; all_thresholds={}; rocs={}; shadow_rows=[]
    for ci,case in enumerate(CASES):
        out=output/case; (out/'shadows').mkdir(parents=True)
        f,_,_=ctx.request(case); in_f=np.zeros(len(tr['y']),bool); in_f[f]=True
        outside_f=set(enc_tr[~in_f]); cohorts={}
        for cohort in ['F','R_group']:
            with np.load(COHORT_SOURCE/f'{case}_{cohort}_original_scores.npz') as z:
                m=np.array([ctx.index[v] for v in z['member_ids']]); n=np.array([tindex[v] for v in z['nonmember_ids']])
            small=unit_size[m]<=MAX_UNIT
            separate=np.array([u not in outside_f for u in enc_tr[m]]) if cohort=='F' else ~np.isin(enc_tr[m],enc_tr[f])
            cohorts[cohort]=(m[small&separate],n)
            eligibility[f'{case}/{cohort}']={'members_before':len(m),'members_eligible':int((small&separate).sum()),
                'excluded_duplicate_outside_forget' if cohort=='F' else 'excluded_unit_touches_forget':int((~separate).sum()),
                'excluded_unit_larger_than_%d'%MAX_UNIT:int((~small&separate).sum()),'nonmembers':len(n)}
        member_units=np.unique(np.concatenate([enc_tr[m] for m,_ in cohorts.values()]))
        ptr=np.flatnonzero(np.isin(enc_tr,member_units)); pte=np.unique(np.concatenate([n for _,n in cohorts.values()]))
        pool_x=np.r_[tr['X'][ptr],te['X'][pte]]; pool_y=np.r_[tr['y'][ptr],te['y'][pte]]
        pool_units=np.r_[[f'tr{u}' for u in enc_tr[ptr]],[f'te{u}' for u in enc_te[pte]]]
        pool_clusters=np.r_[clusters_tr[ptr],clusters_te[pte]]
        pos_tr={v:i for i,v in enumerate(ptr)}; pos_te={v:len(ptr)+i for i,v in enumerate(pte)}
        masks=balanced_masks(n_shadows,pool_units,seed+4000+ci)
        test_client=np.random.default_rng(seed+4500+ci).integers(0,len(ctx.parts),len(pte))
        base=np.ones(len(tr['y']),bool); base[ptr]=False
        np.savez_compressed(out/'pool.npz',train_row_ids=tr['row_id'][ptr],test_row_ids=te['row_id'][pte],
                            units=pool_units,test_client=test_client,masks=masks)
        signals=np.empty((n_shadows,len(pool_y)))
        for k in range(n_shadows):
            member=base.copy(); member[ptr[masks[k,:len(ptr)]]]=True; rows=np.flatnonzero(member)
            extra=masks[k,len(ptr):]; workers=[]
            for j in range(len(ctx.parts)):
                a=rows[client_of[rows]==j]; b=pte[extra&(test_client==j)]
                workers.append(ClientWorker(j,np.r_[tr['X'][a],te['X'][b]],np.r_[tr['y'][a],te['y'][b]],np.r_[tr['row_id'][a],te['row_id'][b]]))
            s=seed+5000+100*ci+k
            m,logs=fedavg(model(tr['X'].shape[1],cfg['hidden'],s),workers,cfg['fl'],cfg['fl']['rounds'],s)
            signals[k]=lira_signal(m,pool_x,pool_y)
            torch.save({'state_dict':state(m),'seed':s,'fl_config':cfg['fl']},out/'shadows'/f'shadow{k}.pt')
            shadow_rows.append({'case':case,'shadow':k,'seed':s,'train_rows':len(rows),'test_rows_in':int(extra.sum()),
                'pool_in':int(masks[k].sum()),'optimizer_steps':sum(c['steps'] for r in logs for c in r['clients']),
                'wall_seconds':sum(r['wall_seconds'] for r in logs)})
            if k%8==7: print(f'{case}: shadow {k+1}/{n_shadows}',flush=True)
        # Positive control, paper Sec. V protocol: held-out shadows with known membership.
        thresholds={}; control={}
        for variant,(mode,var) in VARIANTS.items():
            sm,sn,gm,gn,src=[],[],[],[],[]
            for held in range(LOO):
                keep=np.arange(n_shadows)!=held
                s=LiRA(mode,var).fit(signals[keep],masks[keep]).score(signals[held])
                sm.append(s[masks[held]]); sn.append(s[~masks[held]])
                gm.append(pool_clusters[masks[held]]); gn.append(pool_clusters[~masks[held]])
            sm,sn,gm,gn=map(np.concatenate,(sm,sn,gm,gn))
            thresholds[variant]=thresholds_from(sn)
            control[variant]={'case':case,'variant':variant,'primary':variant==PRIMARY,**summarize(sm,sn,gm,gn,thresholds[variant])}
            control[variant]['usable']=usable(control[variant]); loo_rows.append(control[variant])
            rocs[f'{case}/loo/{variant}']=grid_roc(sm,sn)
        all_thresholds[case]=thresholds
        names={**names_base,'alternating_recovery':ctx.selections[case]['chosen']+'_post'}
        target_signals={name:lira_signal(ctx.load(case,ck),pool_x,pool_y) for name,ck in names.items()}
        attacks={v:LiRA(mode,var).fit(signals,masks) for v,(mode,var) in VARIANTS.items()}
        np.savez_compressed(out/'signals.npz',shadows=signals,**{f'target_{k}':v for k,v in target_signals.items()})
        np.savez_compressed(out/'attack_params.npz',**{f'{v}_{p}':getattr(a,p) for v,a in attacks.items()
                            for p in ['in_mean','in_std','out_mean','out_std'] if getattr(a,p) is not None})
        case_rows=[]
        for cohort,(m,n) in cohorts.items():
            mp=np.array([pos_tr[i] for i in m]); npos=np.array([pos_te[i] for i in n])
            for variant,attack in attacks.items():
                score=lambda name,idx: attack.score(target_signals[name][idx],idx)
                ref=(score('retrain',mp),score('retrain',npos)); thr=thresholds[variant]
                ref_summary=score_summary(*ref,thr)
                for name in names:
                    sm,sn=score(name,mp),score(name,npos)
                    row={'case':case,'cohort':cohort,'model':name,'variant':variant,'primary':variant==PRIMARY,
                         **summarize(sm,sn,clusters_tr[m],clusters_te[n],thr,reference=ref)}
                    for metric in ['auc']+[f'tpr_{f}' for f in FPRS]:
                        row[metric+'_minus_retrain']=row[metric]-ref_summary[metric]
                    case_rows.append(row)
                    if variant==PRIMARY: rocs[f'{case}/{cohort}/{name}']=roc_data(sm,sn)
                np.savez_compressed(out/f'{cohort}_{variant}_scores.npz',member_ids=tr['row_id'][m],nonmember_ids=te['row_id'][n],
                    **{f'{name}_{role}':score(name,idx) for name in names for role,idx in [('member',mp),('nonmember',npos)]})
        for row in case_rows:
            r_control=next(r for r in case_rows if r['cohort']=='R_group' and r['model']=='original' and r['variant']==row['variant'])
            row['interpretation']=('EMPIRICAL_ATTACK_DETECTABLE_NOT_DELETION_PROOF' if control[row['variant']]['usable'] and usable(r_control)
                                   else 'MIA_INCONCLUSIVE')
        comparison.extend(case_rows)
        fig,axes=plt.subplots(1,2,figsize=(12,5))
        for ax,cohort in zip(axes,cohorts):
            curve=rocs[f'{case}/loo/{PRIMARY}']; ax.loglog(curve['fpr'],curve['tpr'],'k--',label='LOO shadow control')
            for name in names:
                curve=rocs[f'{case}/{cohort}/{name}']; ax.loglog(curve['fpr'],curve['tpr'],label=name)
            ax.plot([1e-4,1],[1e-4,1],':',color='grey'); ax.set(xlim=(1e-4,1),ylim=(1e-4,1),xlabel='FPR',ylabel='TPR',
                title=f'{cohort} ({PRIMARY})'); ax.legend(fontsize=7)
        fig.suptitle(case); fig.tight_layout(); fig.savefig(output/f'{case}_roc_loglog.png',dpi=160); plt.close(fig)
        a=attacks['online_per_example']; d=np.abs(a.in_mean-a.out_mean)/(a.in_std+a.out_std)
        pick=np.argsort(d[:len(ptr)])[[-1,-2,len(ptr)//2,0]]
        fig,axes=plt.subplots(1,4,figsize=(16,3.5))
        for ax,i in zip(axes,pick):
            ax.hist(signals[masks[:,i],i],bins=15,alpha=.6,color='tab:red',label='IN')
            ax.hist(signals[~masks[:,i],i],bins=15,alpha=.6,color='tab:blue',label='OUT')
            ax.set(title=f'pool {i}; d={d[i]:.2f}',xlabel='logit-scaled confidence'); ax.legend(fontsize=7)
        fig.suptitle(case+'; shadow IN/OUT signals (paper Fig. 3/4)'); fig.tight_layout()
        fig.savefig(output/f'{case}_in_out_hist.png',dpi=160); plt.close(fig)
        print(f'Completed LiRA {case}',flush=True)
    pd.DataFrame(comparison).to_csv(output/'comparison.csv',index=False)
    pd.DataFrame(loo_rows).to_csv(output/'loo_control.csv',index=False)
    pd.DataFrame(shadow_rows).to_csv(output/'shadows.csv',index=False)
    write_json(output/'thresholds.json',all_thresholds); write_json(output/'eligibility.json',eligibility)
    write_json(output/'roc.json',rocs)
    write_json(output/'status.json',{'completed':True,'target_rows':len(comparison),'shadows_per_case':n_shadows,
        'total_seconds':time.perf_counter()-started,'test_used_for_selection':False,'unlearning_success_claimed':False,
        'target_seed_count':1})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--shadows',type=int,default=64)
    a=p.parse_args(); run(a.output,a.shadows)
