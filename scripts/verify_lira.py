"""Independently verify a completed LiRA run: hashes, balanced roles, signals, cohorts and fixed thresholds."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.experiment_context import Context,use_run_device,CASES
from src.data.prepare import connected_groups,digest
from src.eval.lira import lira_signal
from src.fl.fedavg_core import model

def verify(run,checked_shadows=3):
    use_run_device(run)
    hashes=json.loads((run/'artifact_hashes.json').read_text())
    for name,sha in hashes.items(): assert digest(run/name)==sha,name
    status=json.loads((run/'status.json').read_text()); assert status['completed']
    manifest=json.loads((run/'manifest.json').read_text()); protocol=manifest['protocol']; n=protocol['shadows_per_case']
    ctx=Context(); tr,te=ctx.tr,ctx.data['test']; source=Path(protocol['cohort_source'])
    enc=connected_groups([row.tobytes() for row in tr['X']]); tindex={v:i for i,v in enumerate(te['row_id'])}
    df=pd.read_csv(run/'comparison.csv'); shadows=pd.read_csv(run/'shadows.csv')
    assert len(df)==status['target_rows'] and not df.select_dtypes('number').isna().any().any()
    thresholds=json.loads((run/'thresholds.json').read_text()); recomputed=0
    for case in CASES:
        with np.load(run/case/'pool.npz') as z: pool={k:z[k] for k in z.files}
        masks=pool['masks']; assert masks.shape[0]==n and (masks.sum(0)==n//2).all()
        for u in np.unique(pool['units']):
            cols=masks[:,pool['units']==u]; assert (cols==cols[:,:1]).all(),u
        ptr=np.array([ctx.index[v] for v in pool['train_row_ids']]); pte=np.array([tindex[v] for v in pool['test_row_ids']])
        # Pool train rows are complete duplicate units: no copy of a toggled point stays always-IN.
        assert np.isin(enc,enc[ptr]).sum()==len(ptr)
        f,_,_=ctx.request(case); in_f=np.zeros(len(enc),bool); in_f[f]=True
        rows=shadows[shadows.case==case].sort_values('shadow')
        base=len(tr['y'])-len(ptr)
        np.testing.assert_array_equal(rows.train_rows,base+masks[:,:len(ptr)].sum(1))
        np.testing.assert_array_equal(rows.test_rows_in,masks[:,len(ptr):].sum(1))
        with np.load(run/case/'signals.npz') as z: signals=z['shadows']; targets={k:z[k] for k in z.files if k.startswith('target_')}
        assert np.isfinite(signals).all() and all(np.isfinite(v).all() for v in targets.values())
        x=np.r_[tr['X'][ptr],te['X'][pte]]; y=np.r_[tr['y'][ptr],te['y'][pte]]
        for k in np.random.default_rng(0).choice(n,checked_shadows,replace=False):
            ck=torch.load(run/case/'shadows'/f'shadow{k}.pt',weights_only=True)
            m=model(x.shape[1],ctx.cfg['hidden'],int(ck['seed'])); m.load_state_dict(ck['state_dict'])
            np.testing.assert_allclose(lira_signal(m,x,y),signals[k],atol=1e-9); recomputed+=1
        np.testing.assert_allclose(lira_signal(ctx.load(case,'original'),x,y),targets['target_original'],atol=1e-9)
        for cohort in ['F','R_group']:
            with np.load(source/f'{case}_{cohort}_original_scores.npz') as z: old_m,old_n=z['member_ids'],z['nonmember_ids']
            for variant,thr in thresholds[case].items():
                with np.load(run/case/f'{cohort}_{variant}_scores.npz') as z: mids,nids=z['member_ids'],z['nonmember_ids']
                assert set(mids)<=set(old_m); np.testing.assert_array_equal(nids,old_n)
                rows=df[(df.case==case)&(df.cohort==cohort)&(df.variant==variant)]
                for fpr,t in thr.items():
                    assert rows[f'threshold_{fpr}'].nunique()==1 and np.allclose(rows[f'threshold_{fpr}'],t,rtol=1e-12,atol=0)
            mi=np.array([ctx.index[v] for v in mids])
            if cohort=='F': assert not set(enc[mi])&set(enc[~in_f]),'F member duplicated outside forget set'
            else: assert not set(enc[mi])&set(enc[f]),'R member shares a unit with F'
    for _,r in df[df.model=='retrain'].iterrows():
        assert r.auc_minus_retrain==0
        if r.ci_status=='GROUP_BOOTSTRAP_95_PERCENT': assert r.auc_minus_retrain_ci_low==r.auc_minus_retrain_ci_high==0
    return {'verified_artifacts':len(hashes),'comparison_rows':len(df),'shadows_per_case':n,'balanced_masks':True,
            'recomputed_shadow_signals':recomputed,'cohorts_subset_of_v3':True,'thresholds_fixed_across_models':True,
            'retrain_self_difference_zero':True}

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True)
    print(json.dumps(verify(p.parse_args().run),indent=2))
