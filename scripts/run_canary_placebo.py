"""Placebo (permutation) test of the canary membership statistic. Post hoc: added after the canary-LiRA run.

Canary sources and matched nonmembers are uniform draws from the same pool, so without a membership effect their
labels are exchangeable. The canary MIA's statistic, the mean over seeds of AUC(model) - AUC(retrain) on one fixed
cohort, is recomputed under random relabellings of that cohort. Its spread is what model differences alone produce
on a fixed cohort; the paired CI over seeds cannot see it, because every seed reuses the same cohort.
"""
import argparse
import json
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
from scipy.stats import rankdata
from scripts.experiment_context import Context,finalize
from scripts.run_canary_lira import MODELS
from src.data.prepare import digest,write_json

LIRA=Path('runs/canary_lira_seed42'); CANARY_MIA=Path('runs/canary_mia_seed42')
CONFIG={'permutations':2000,'seed':0,
        'levels':{'row':'canary_mia cohorts: 500 canary rows vs 500 matched nonmember rows',
                  'unit':'canary_lira cohorts: member-only vs nonmember-only feature vectors'},
        'score':'phi = z_normal - z_attack from canary_lira signals (monotone in the canary-MIA score log p(normal))',
        'statistic':'mean over seeds of AUC(model) - AUC(retrain) on the cohort',
        'p_value':'two-sided: (#{|null| >= |observed|} + 1) / (permutations + 1)'}

def aucs(scores,labels):
    """AUC of every score vector [k, n] under every label vector [p, n] with equal member counts; ties get average ranks."""
    ranks=np.stack([rankdata(v) for v in scores]); n1=int(labels[0].sum()); n0=labels.shape[1]-n1
    return (labels.astype(float)@ranks.T-n1*(n1+1)/2)/(n1*n0)

def placebo(scores,keys,labels,permutations,rng):
    """{model: (observed, null)} for the mean over seeds of AUC(model) - AUC(retrain); keys are (seed, model) per score row."""
    A=aucs(scores,np.stack([labels]+[rng.permutation(labels) for _ in range(permutations)]))
    index={k:i for i,k in enumerate(keys)}; seeds=sorted({s for s,_ in keys}); out={}
    for m in [m for m in MODELS if m!='retrain' and (seeds[0],m) in index]:
        d=np.mean([A[:,index[(s,m)]]-A[:,index[(s,'retrain')]] for s in seeds],axis=0); out[m]=(float(d[0]),d[1:])
    return out

def run(output):
    for root in [LIRA,CANARY_MIA]:
        for name,sha in json.loads((root/'artifact_hashes.json').read_text()).items():
            if digest(root/name)!=sha: raise ValueError(f'Changed source artifact {root/name}')
    started=time.perf_counter(); ctx=Context()
    ctx.manifest_for(output,'canary_placebo',{**CONFIG,'post_hoc':True,
        'sources':{str(r):digest(r/'artifact_hashes.json') for r in [LIRA,CANARY_MIA]}})
    rng=np.random.default_rng(CONFIG['seed']); mia=pd.read_csv(CANARY_MIA/'mia_rows.csv'); rows=[]; check={}
    for name in json.loads((LIRA/'status.json').read_text())['requests']:
        with np.load(LIRA/name/'pool.npz') as z: unit,role,canaries=z['unit'],z['role'],len(z['source_row_ids'])
        with np.load(LIRA/name/'signals.npz') as z: keys=list(zip(z['target_seeds'].tolist(),z['target_models'].tolist())); T=z['targets']
        keep=role!='shared'
        cohorts={'row':(T[:,unit],np.arange(len(unit))<canaries),'unit':(T[:,keep],role[keep]=='member')}
        ref=mia[mia.request==name].set_index(['seed','model']).auc
        check[name]=float(np.abs(aucs(*cohorts['row'][:1],cohorts['row'][1][None])[0]-np.array([ref[k] for k in keys])).max())
        if check[name]>1e-3: raise AssertionError(f'{name}: row-level AUCs do not reproduce the canary MIA')
        for level,(scores,labels) in cohorts.items():
            for m,(obs,null) in placebo(scores,keys,labels,CONFIG['permutations'],rng).items():
                rows.append({'request':name,'level':level,'model':m,'members':int(labels.sum()),'nonmembers':int((~labels).sum()),
                             'observed':obs,'null_mean':null.mean(),'null_sd':null.std(ddof=1),
                             'null_q025':np.quantile(null,.025),'null_q975':np.quantile(null,.975),
                             'p_value':(np.sum(np.abs(null)>=abs(obs))+1)/(len(null)+1)})
    pd.DataFrame(rows).to_csv(output/'placebo.csv',index=False)
    write_json(output/'status.json',{'completed':True,'post_hoc':True,'max_abs_row_auc_diff_vs_canary_mia':check,
        'note':'differences up to 1e-3 come from ties in the saturated log p(normal) of the old score',
        'total_seconds':time.perf_counter()-started})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
