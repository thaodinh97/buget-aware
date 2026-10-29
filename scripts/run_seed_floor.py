"""Seed-noise floor: judge FU checkpoints against clouds of independently seeded retrains and originals."""
import argparse
import sys
import time
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scipy.special import expit
from scripts.experiment_context import Context,finalize
from src.data.prepare import write_json
from src.eval.metrics import utility
from src.eval.membership_validation import logit_margin,stable_baselines
from src.eval.seed_floor import distance_profile,classify
from src.fl.fedavg_core import model,state,fedavg,ClientWorker,get_device

MODELS=['original','retrain','no_update_post','ga_only_post','gd_only_post','alt_3_1_post','alt_1_1_post',
        'alt_1_3_post','ga_gd_with_kd_post','conpfu_inspired_post','retain_kd_matched_post']

def train(ctx,partitions,seed):
    tr=ctx.tr; workers=[ClientWorker(j,tr['X'][idx],tr['y'][idx],tr['row_id'][idx]) for j,idx in enumerate(partitions)]
    m,_=fedavg(model(tr['X'].shape[1],ctx.cfg['hidden'],seed),workers,ctx.cfg['fl'],ctx.cfg['fl']['rounds'],seed)
    return m

def reproduction(fresh,cached):
    a,b=state(fresh),state(cached)
    return {'exact':all(torch.equal(a[k],b[k]) for k in a),'max_abs_diff':max(float((a[k]-b[k]).abs().max()) for k in a)}

def run(output,n_seeds):
    ctx=Context(); tr,te=ctx.tr,ctx.data['test']; seed=ctx.cfg['seed']; started=time.perf_counter()
    seeds=[seed+10000+i for i in range(n_seeds)]
    print(f'Using device: {get_device()}',flush=True)
    ctx.manifest_for(output,'seed_floor',{'seeds':seeds,'models':MODELS,'requests':list(ctx.selections),
        'clouds':'R_t = retrain on D\\F with seed t; O_t = original on D with the same seed t (init and batch order)',
        'distance':'mean |p_attack difference| to the retrain cloud; same-seed O/R pairs excluded',
        'scalars':['ce_F','macro_f1_test'],'envelope':'95% normal prediction interval per cloud; separable iff the two intervals do not overlap',
        'selection':'descriptive; no FU setting chosen from these results'})
    (output/'checkpoints').mkdir()
    margin=lambda m:(logit_margin(m,tr['X']),logit_margin(m,te['X']))
    checks={'original':reproduction(train(ctx,ctx.parts,seed),ctx.load(next(iter(ctx.selections)),'original'))}
    originals=[]
    for s in seeds:
        m=train(ctx,ctx.parts,s); torch.save({'state_dict':state(m),'seed':s},output/'checkpoints'/f'original_seed{s}.pt')
        originals.append(margin(m))
    print(f'Trained {n_seeds} original seeds',flush=True)
    rows=[]; clouds=[]; paired=[]
    for case,selection in ctx.selections.items():
        f,retained,_=ctx.request(case)
        checks[case]=reproduction(train(ctx,retained,seed),ctx.load(case,'retrain'))
        retrains=[]
        for s in seeds:
            m=train(ctx,retained,s); torch.save({'state_dict':state(m),'seed':s},output/'checkpoints'/f'{case}_retrain_seed{s}.pt')
            retrains.append(margin(m))
        targets={name:margin(ctx.load(case,name)) for name in MODELS}
        sets={'F':(tr['y'][f],lambda mg:mg[0][f]),'test':(te['y'],lambda mg:mg[1])}
        def stats(mg):
            return {'ce_F':float(-stable_baselines(sets['F'][0],sets['F'][1](mg))['negative_loss'].mean()),
                    'macro_f1_test':float(utility(te['y'],expit(sets['test'][1](mg)))['macro_f1'])}
        prob=lambda group,key:np.array([expit(sets[key][1](mg)) for mg in group])
        values={}
        for key in sets:
            prof=distance_profile(prob(retrains,key),prob(originals,key),prob(targets.values(),key))
            values[f'mae_{key}']=(prof['retrain'],prof['original'],dict(zip(MODELS,prof['models'])))
            for i,s in enumerate(seeds): paired.append({'case':case,'seed':s,'set':key,'same_seed_original_retrain_mae':prof['paired_same_seed'][i]})
        r_stats=[stats(mg) for mg in retrains]; o_stats=[stats(mg) for mg in originals]; t_stats={n:stats(mg) for n,mg in targets.items()}
        for key in ['ce_F','macro_f1_test']:
            values[key]=(np.array([v[key] for v in r_stats]),np.array([v[key] for v in o_stats]),{n:v[key] for n,v in t_stats.items()})
        np.savez_compressed(output/f'{case}_clouds.npz',seeds=seeds,
            **{f'{metric}_{cloud}':v for metric,(r,o,_) in values.items() for cloud,v in [('retrain',r),('original',o)]})
        for metric,(r,o,per_model) in values.items():
            clouds.extend({'case':case,'metric':metric,'cloud':cloud,'seed':s,'value':float(v)}
                          for cloud,vals in [('retrain',r),('original',o)] for s,v in zip(seeds,vals))
            for name,v in per_model.items():
                rows.append({'case':case,'model':name,'selected_fu':name==selection['chosen']+'_post','metric':metric,**classify(v,r,o)})
        print(f'Completed {case}',flush=True)
    pd.DataFrame(rows).to_csv(output/'placement.csv',index=False)
    pd.DataFrame(clouds).to_csv(output/'clouds.csv',index=False)
    pd.DataFrame(paired).to_csv(output/'paired_same_seed.csv',index=False)
    write_json(output/'reproduction.json',checks)
    write_json(output/'status.json',{'completed':True,'seeds':n_seeds,'placement_rows':len(rows),
        'all_seed42_reproduced_exactly':all(c['exact'] for c in checks.values()),
        'total_seconds':time.perf_counter()-started,'unlearning_success_claimed':False})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--output',type=Path,required=True)
    p.add_argument('--seeds',type=int,default=10)
    a=p.parse_args(); run(a.output,a.seeds)
