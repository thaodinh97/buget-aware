"""Verify completed trajectories, attack roles and fixed comparison artifacts."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from src.data.prepare import digest

def verify(recovery,mia):
    totals={}
    for label,path in [('recovery',recovery),('mia',mia)]:
        hashes=json.loads((path/'artifact_hashes.json').read_text())
        for name,sha in hashes.items(): assert digest(path/name)==sha,name
        assert json.loads((path/'status.json').read_text())['completed']
        totals[label+'_verified_artifacts']=len(hashes)
    g=pd.read_csv(recovery/'global.csv'); c=pd.read_csv(recovery/'clients.csv')
    assert len(g)==80 and len(c)==320
    for _,group in g.groupby(['case','start','lr_multiplier']): assert list(group['round'])==list(range(5))
    np.testing.assert_allclose(g.distance_after**2-g.distance_before**2,g.squared_distance_change,atol=1e-10)
    for key,clients in c.groupby(['case','start','lr_multiplier','round']):
        row=g[(g.case==key[0])&(g.start==key[1])&(g.lr_multiplier==key[2])&(g['round']==key[3])].iloc[0]
        np.testing.assert_allclose(clients.weight,clients.retained_count/clients.retained_count.sum())
        np.testing.assert_allclose((clients.weight*clients.dot_to_retrain).sum(),row.dot_to_retrain,atol=1e-5)
        assert clients.optimizer_steps.sum()==row.optimizer_steps
    checks=json.loads((recovery/'checks.json').read_text()); assert len(checks)==8
    assert all(r['round2_matches_cached'] for r in checks)
    with np.load(mia/'roles.npz') as z:
        sets=[set(z[k]) for k in z.files]
        for i,s in enumerate(sets):
            for t in sets[:i]: assert not s&t
    with np.load(mia/'role_components.npz') as z:
        sets=[set(z[k]) for k in z.files]
        for i,s in enumerate(sets):
            for t in sets[:i]: assert not s&t
    df=pd.read_csv(mia/'comparison.csv'); assert len(df)==108
    for _,r in df[df.model=='retrain'].iterrows():
        assert r.auc_minus_retrain==0
        if r.ci_status=='GROUP_BOOTSTRAP_95_PERCENT':
            assert r.auc_minus_retrain_ci_low==r.auc_minus_retrain_ci_high==0
    for (case,cohort),rows in df.groupby(['case','cohort']):
        expected=None
        for name in rows.model.unique():
            with np.load(mia/f'{case}_{cohort}_{name}_scores.npz') as z:
                ids=(z['member_ids'],z['nonmember_ids'])
                if expected is not None:
                    for a,b in zip(ids,expected): np.testing.assert_array_equal(a,b)
                expected=tuple(a.copy() for a in ids)
        for _,r in rows.groupby('attack'):
            assert r['threshold_0.01'].nunique()==1 and r['threshold_0.05'].nunique()==1
    return {**totals,'global_rows':len(g),'client_rows':len(c),'mia_rows':len(df),
            'cached_round2_exact':8,'cohorts_and_thresholds_fixed':True,'geometry_decomposition_verified':True}

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--recovery',type=Path,required=True); p.add_argument('--mia',type=Path,required=True)
    a=p.parse_args(); print(json.dumps(verify(a.recovery,a.mia),indent=2))
