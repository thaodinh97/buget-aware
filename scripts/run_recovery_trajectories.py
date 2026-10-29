"""Fixed recovery trajectories; retraining and test scores are offline diagnostics only."""
import argparse
import json
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.metrics import log_loss
from scripts.experiment_context import Context,CASES,finalize
from src.fl.fedavg_core import state,predict,fedavg,ClientWorker
from src.fu.diagnostics import tensor_hash
from src.eval.metrics import utility
from src.eval.recovery_analysis import alignment,vector,calibration
from src.data.prepare import write_json

def run(output):
    ctx=Context(); cfg=ctx.cfg; tr=ctx.tr
    ctx.manifest_for(output,'recovery_trajectories',{'cases':CASES,'rounds':[0,1,2,3,4],
        'lr_multipliers':[1.,.25],'selection':'none; historical alternating oracle diagnosis only'})
    (output/'checkpoints').mkdir(); rows=[]; clients=[]; checks=[]
    for case in CASES:
        f,retained,request=ctx.request(case); r=np.concatenate(retained)
        workers=[ClientWorker(i,tr['X'][idx],tr['y'][idx],tr['row_id'][idx]) for i,idx in enumerate(retained)]
        orig=ctx.load(case,'original'); retrain=ctx.load(case,'retrain')
        ov,rv=vector(state(orig)),vector(state(retrain))
        ref={s:predict(retrain,ctx.data[s]['X']) for s in ['validation','test','train']}
        chosen=ctx.selections[case]['chosen']
        for start,checkpoint in [('original','original'),('ga_only','ga_only_pre'),('alternating',chosen+'_pre'),('gd_only','gd_only_pre')]:
            for mult in [1.,.25]:
                initial=ctx.load(case,checkpoint); current=ctx.load(case,checkpoint); previous_mae=None
                label=f'{case}_{start}_lr{mult}'
                def observe(round_id,before,after,updates,counts,log):
                    nonlocal previous_mae
                    current.load_state_dict(after); av=vector(after)
                    p={s:predict(current,ctx.data[s]['X']) for s in ['validation','test','train']}
                    u=utility(ctx.data['validation']['y'],p['validation'])
                    row={'case':case,'start':start,'historical_alternating':chosen,'lr_multiplier':mult,'round':round_id+1,
                         **{f'validation_{k}':u[k] for k in ['macro_f1','pr_auc_ap','fpr','log_loss']},
                         **{f'validation_{k}':v for k,v in calibration(ctx.data['validation']['y'],p['validation']).items()},
                         'forget_ce':log_loss(tr['y'][f],p['train'][f],labels=[0,1]),
                         'retain_ce':log_loss(tr['y'][r],p['train'][r],labels=[0,1]),
                         'mae_validation':float(abs(p['validation']-ref['validation']).mean()),
                         'mae_test':float(abs(p['test']-ref['test']).mean()),
                         'mae_forget':float(abs(p['train'][f]-ref['train'][f]).mean()),
                         'l2_original':(av-ov).norm().item(),'l2_retrain':(av-rv).norm().item(),
                         'runtime_seconds':log['wall_seconds'] if log else 0.,
                         'optimizer_steps':sum(c['steps'] for c in log['clients']) if log else 0}
                    row.update(alignment(vector(before),av,rv))
                    row['mae_validation_before']=previous_mae; previous_mae=row['mae_validation']
                    rows.append(row)
                    for update,count,local in zip(updates,counts,log['clients'] if log else []):
                        client=local['client_id']; idx=retained[client]
                        clients.append({'case':case,'start':start,'lr_multiplier':mult,'round':round_id+1,
                            'client_id':client,'retained_count':count,'weight':count/sum(counts),
                            'normal_count':int((tr['y'][idx]==0).sum()),'attack_count':int((tr['y'][idx]==1).sum()),
                            'optimizer_steps':local['steps'],'runtime_seconds':local['compute_seconds'],
                            **alignment(vector(before),vector(update),rv)})
                    torch.save({'state_dict':after,'round':round_id+1,'start':start,'tensor_sha256':tensor_hash(current)},
                               output/'checkpoints'/f'{label}_r{round_id+1}.pt')
                    if round_id==1 and mult==1:
                        cached='no_update_post' if start=='original' else (chosen+'_post' if start=='alternating' else start+'_post')
                        exact=tensor_hash(ctx.load(case,cached))==tensor_hash(current)
                        checks.append({'case':case,'start':start,'round2_matches_cached':exact})
                        if not exact: raise AssertionError('Recovery trajectory differs from cached round2')
                    pd.DataFrame(rows).to_csv(output/'global.csv',index=False)
                    pd.DataFrame(clients).to_csv(output/'clients.csv',index=False)
                observe(-1,state(initial),state(initial),[],[],None)
                fl={**cfg['fl'],'lr':cfg['fl']['lr']*mult}
                fedavg(initial,workers,fl,4,cfg['seed']+900000,observer=observe)
        print(f'Finished recovery {case}',flush=True)
    write_json(output/'checks.json',checks)
    df=pd.DataFrame(rows)
    for case in CASES:
        fig,axes=plt.subplots(2,2,figsize=(11,8))
        for (start,mult),g in df[df.case==case].groupby(['start','lr_multiplier']):
            for ax,field in zip(axes.flat,['mae_validation','mae_test','validation_log_loss','l2_retrain']):
                ax.plot(g['round'],g[field],marker='o',ls='-' if mult==1 else '--',label=f'{start}; lr x{mult}')
                ax.set(xlabel='Recovery round',ylabel=field,xticks=range(5)); ax.grid(alpha=.2)
        axes[0,0].legend(fontsize=7); fig.suptitle(case+' (offline diagnostics)'); fig.tight_layout()
        fig.savefig(output/f'{case}.png',dpi=160); plt.close(fig)
    write_json(output/'status.json',{'completed':True,'global_rows':len(rows),'client_rows':len(clients),
                                    'test_used_for_selection':False,'dt_implemented':False})
    finalize(output)

if __name__=='__main__':
    p=argparse.ArgumentParser(); p.add_argument('--output',type=Path,required=True)
    run(p.parse_args().output)
