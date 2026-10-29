"""Frozen-artifact FU diagnostics: no split regeneration, no test-based tuning."""
import argparse
import copy
import json
import sys
import time
import traceback
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.run_pilot import load_data
from scripts.verify_run import verify
from src.data.prepare import digest,write_json,signatures,GROUP_COLUMNS,overlaps
from src.fl.fedavg_core import model,state,ClientWorker,fedavg,predict,get_device
from src.fu.conpfu_inspired import alternating,distill
from src.fu.diagnostics import Observer,tensor_hash
from src.eval.metrics import utility,similarity,mia,calibrate_mia


def family_match(member_indices, train, test, types, seed):
    rng=np.random.default_rng(seed); mi=[]; ni=[]; coverage=[]
    for label in [0,1]:
        for family in sorted(set(types['train'][member_indices])):
            m=member_indices[(train['y'][member_indices]==label)&(types['train'][member_indices]==family)]
            n=np.flatnonzero((test['y']==label)&(types['test']==family))
            if not len(m): continue
            count=min(len(m),len(n),500)
            coverage.append({'label':label,'type':family,'available_members':len(m),'available_nonmembers':len(n),'selected_each':count})
            mi.extend(rng.permutation(m)[:count]); ni.extend(rng.permutation(n)[:count])
    if not mi: raise ValueError('No family-matched MIA support')
    return np.array(mi,dtype=int),np.array(ni,dtype=int),coverage


def run(source,data_path,output):
    if output.exists(): raise FileExistsError(output)
    verify(source,data_path)
    manifest=json.loads((source/'manifest.json').read_text()); cfg=manifest['config']
    if manifest['smoke_only']: raise ValueError('Controlled pilot requires full original artifacts')
    torch.set_num_threads(cfg['threads']); torch.use_deterministic_algorithms(True)
    device=get_device()
    print(f'Using device: {device}',flush=True)
    data=load_data(data_path); train=data['train']; test=data['test']
    metadata=pd.read_csv(data_path/'metadata.csv',dtype=str).set_index('row_id')
    types={s:metadata.loc[d['row_id'],'type'].to_numpy() for s,d in data.items()}
    # Independently verify frozen raw bidirectional grouping without rewriting artifacts.
    raw=pd.read_csv('dataset/train_test_network.csv',low_memory=False)
    raw_hash=digest('dataset/train_test_network.csv')
    if raw_hash!=json.loads((data_path/'manifest.json').read_text())['dataset_sha256']: raise ValueError('Raw hash mismatch')
    forward=signatures(raw[GROUP_COLUMNS]); rev=raw[GROUP_COLUMNS].copy()
    rev.columns=['dst_ip','dst_port','src_ip','src_port','proto']
    reverse=signatures(rev[GROUP_COLUMNS]); bidir=np.array([min(a,b) for a,b in zip(forward,reverse)])
    keys=[]; splits=[]
    for s,d in data.items():
        keys.extend(bidir[[int(i.rsplit(':',1)[1]) for i in d['row_id']]])
        splits.extend([s]*len(d['y']))
    audited=overlaps(np.array(keys),np.array(splits))
    if any(audited.values()): raise ValueError('Frozen bidirectional leakage detected')
    index={v:i for i,v in enumerate(train['row_id'])}
    with np.load(source/'partitions.npz',allow_pickle=False) as arrays:
        parts=[np.array([index[v] for v in arrays[f'client_{i}']]) for i in range(5)]
    def load(name):
        m=model(train['X'].shape[1],cfg['hidden'],cfg['seed'])
        saved=torch.load(source/'checkpoints'/f'{name}.pt',weights_only=True)
        if saved['manifest_sha256']!=digest(source/'manifest.json'): raise ValueError('Checkpoint manifest mismatch')
        m.load_state_dict(saved['state_dict']); return m
    original=load('original')
    output.mkdir(parents=True); (output/'checkpoints').mkdir()
    write_json(output/'manifest.json',{'source_run':str(source),'source_manifest_sha256':digest(source/'manifest.json'),
               'data_manifest_sha256':digest(data_path/'manifest.json'),'config':cfg,'device':str(device),
               'frozen_splits_requests':True,
               'sources':{str(p):digest(p) for root in ['src','scripts'] for p in Path(root).rglob('*.py')},
               'shadow_models_trained':0,'mia_score':'negative CE; never flip by observed AUC',
               'conpfu_baseline':'global-start/full-local-KD adaptation; no last-round subtraction/incentives; common recovery',
               'runtime_scope':'new stage optimizer time and observation overhead separated; cached references runtime null'})
    write_json(output/'frozen_data_audit.json',{'bidirectional_overlap':audited,'split_changed':False})
    historical={}
    old_times=json.loads((source/'timings.json').read_text())
    for label,log_name,time_key in [('original','original_training.json','original_seconds')]+[
            (f'{f}_retrain',f'{f}_retrain_training.json',f'{f}_retrain_seconds') for f in cfg['fractions']]:
        logs=json.loads((source/log_name).read_text())
        historical[label]={'historical_seconds':old_times[time_key],
                           'payload_bytes_up':sum(e['payload_bytes_up'] for e in logs),
                           'payload_bytes_down':sum(e['payload_bytes_down'] for e in logs),
                           'optimizer_steps':sum(c['steps'] for e in logs for c in e['clients']),
                           'provenance':str(source/log_name),'warning':'cached historical run; warmup/order differ, not matched speedup measurement'}
    write_json(output/'historical_reference_costs.json',historical)
    thresholds=calibrate_mia(data['calibration']['y'],predict(original,data['calibration']['X']))
    write_json(output/'mia_protocol.json',{'thresholds':thresholds,'fit_split':'calibration','score_direction':'higher is member',
               'shadow_training':'not implemented; fixed score does not require shadow models',
               'protocols':['exact legacy evaluation IDs','additional label+type matched cohort'],
               'warning':'family matching is a sensitivity analysis with different coverage; no group-characteristic matching or causal membership guarantee'})
    rows=[]; details={}; allcoverage=[]; started=time.perf_counter()
    methods=['original','no_op','retrain','retain_only','retain_gd','fu_no_kd','fu_retain_kd','conpfu_inspired_baseline','conpfu_local_only']
    for fraction in cfg['fractions']:
        for method in methods: allcoverage.append({'fraction':fraction,'method':method,'status':'not_run'})
    try:
        for fraction in cfg['fractions']:
            name=str(fraction); req=json.loads((source/f'request_{name}.json').read_text()); target=req['target']
            f=np.array([index[v] for v in req['forget_row_ids']]); r=np.array([index[v] for v in req['retain_row_ids']])
            retained=[np.setdiff1d(p,f) for p in parts]; ri=np.concatenate(retained)
            if set(f)&set(ri): raise AssertionError('F in recovery')
            sig=metadata.loc[train['row_id'],'feature_signature'].to_numpy()
            global_signatures=set(sig[ri]); local_signatures=set(sig[r])
            write_json(output/f'{name}_request_audit.json',{'forget_count':len(f),'target_retain_count':len(r),
                       'forget_class_counts':np.bincount(train['y'][f],minlength=2).tolist(),
                       'retain_class_counts':np.bincount(train['y'][r],minlength=2).tolist(),
                       'F_fraction_with_equivalent_global_R':sum(s in global_signatures for s in sig[f])/len(f),
                       'F_fraction_with_equivalent_local_R':sum(s in local_signatures for s in sig[f])/len(f),
                       'request_sha256':digest(source/f'request_{name}.json')})
            def worker(i,idx): return ClientWorker(i,train['X'][idx],train['y'][idx],train['row_id'][idx])
            fw,rw=worker(target,f),worker(target,r)
            recovery_workers=[worker(i,idx) for i,idx in enumerate(retained)]
            models={'original':original,'no_op':copy.deepcopy(original),'retrain':load(f'{name}_retrain')}
            lineage={m:{'cached_source':str(source/'checkpoints'/('original.pt' if m in ['original','no_op'] else f'{name}_retrain.pt'))} for m in models}
            costs={m:{'optimizer_seconds':None,'diagnostic_seconds':0.,'payload_bytes_up':0,'payload_bytes_down':0,'optimizer_steps':0,'cached':True} for m in models}
            local_results={}; local_costs={}
            for branch,skip in [('fu',False),('retain_gd',True)]:
                obs=Observer(output,f'{name}_{branch}',original,fw,rw); t=time.perf_counter()
                local,logs=alternating(original,fw,rw,cfg['fu'],cfg['seed'],observer=obs,skip_ga=skip)
                elapsed=time.perf_counter()-t
                local_results[branch]=local
                local_costs[branch]={'optimizer_seconds':max(0.,elapsed-obs.seconds),'diagnostic_seconds':obs.seconds,
                                     'optimizer_steps':sum(l['optimizer_steps'] for l in logs)}
                write_json(output/f'{name}_{branch}_steps.json',logs)
            kd_models={}; kd_cost={}
            for branch,w in [('retain',rw),('full_local',worker(target,parts[target]))]:
                obs=Observer(output,f'{name}_kd_{branch}',original,fw,rw); t=time.perf_counter()
                kd,losses=distill(original,local_results['fu'],w,cfg['fu'],cfg['seed'],observer=obs)
                kd_models[branch]=kd
                kd_cost[branch]={'optimizer_seconds':max(0.,time.perf_counter()-t-obs.seconds),'diagnostic_seconds':obs.seconds,'optimizer_steps':len(losses)}
                write_json(output/f'{name}_kd_{branch}_losses.json',losses)
                write_json(output/f'{name}_kd_{branch}_lineage.json',{'teacher_tensor_sha256':tensor_hash(local_results['fu']),
                           'student_initial_tensor_sha256':tensor_hash(original),'training_data':'R' if branch=='retain' else 'F_union_R',
                           'forgotten_data_allowed_only_in_local_operation':True})
            models['conpfu_local_only']=kd_models['full_local']
            lineage['conpfu_local_only']={'after':'full_local_kd','recovery_rounds':0}
            size=sum(v.numel()*v.element_size() for v in state(original).values())
            for method,start,lc,kc in [('retain_only',original,{},{}),('retain_gd',local_results['retain_gd'],local_costs['retain_gd'],{}),
                    ('fu_no_kd',local_results['fu'],local_costs['fu'],{}),('fu_retain_kd',kd_models['retain'],local_costs['fu'],kd_cost['retain']),
                    ('conpfu_inspired_baseline',kd_models['full_local'],local_costs['fu'],kd_cost['full_local'])]:
                t=time.perf_counter(); final,logs=fedavg(start,recovery_workers,cfg['fl'],cfg['recovery_rounds'],cfg['seed']+900000)
                models[method]=final
                lineage[method]={'recovery_start_tensor_sha256':tensor_hash(start),'recovery_rounds':cfg['recovery_rounds'],
                                 'recovery_seed':cfg['seed']+900000,'counts':[len(w.y) for w in recovery_workers]}
                costs[method]={'optimizer_seconds':time.perf_counter()-t+lc.get('optimizer_seconds',0)+kc.get('optimizer_seconds',0),
                               'diagnostic_seconds':lc.get('diagnostic_seconds',0)+kc.get('diagnostic_seconds',0),
                               'optimizer_steps':sum(c['steps'] for entry in logs for c in entry['clients'])+lc.get('optimizer_steps',0)+kc.get('optimizer_steps',0),
                               'payload_bytes_up':sum(e['payload_bytes_up'] for e in logs)+(size if lc else 0),
                               'payload_bytes_down':sum(e['payload_bytes_down'] for e in logs)+(size if lc else 0),'cached':False}
                write_json(output/f'{name}_{method}_recovery.json',logs)
            costs['conpfu_local_only']={k:local_costs['fu'][k]+kd_cost['full_local'][k] for k in local_costs['fu']}
            costs['conpfu_local_only'].update(payload_bytes_up=size,payload_bytes_down=size,cached=False)
            # All evaluation cohorts fixed once, shared by every compared model.
            test_index={v:i for i,v in enumerate(test['row_id'])}
            with np.load(source/f'{name}_evaluation_ids.npz',allow_pickle=False) as ev:
                cohorts={
                    'legacy_F':(np.array([index[i] for i in ev['forget']]),np.array([test_index[i] for i in ev['forget_nonmember']])),
                    'legacy_R':(np.array([index[i] for i in ev['retained']]),np.array([test_index[i] for i in ev['retained_nonmember']]))}
            for label,pool in [('F',f),('R',ri[~np.isin(train['group_id'][ri],train['group_id'][f])])]:
                mi,ni,coverage=family_match(pool,train,test,types,cfg['seed'])
                cohorts[f'family_{label}']=(mi,ni)
                write_json(output/f'{name}_mia_{label}_coverage.json',coverage)
            np.savez_compressed(output/f'{name}_mia_ids.npz',**{f'{k}_{side}':(train if side=='member' else test)['row_id'][idx]
                                for k,pair in cohorts.items() for side,idx in zip(['member','nonmember'],pair)})
            reftrain=predict(models['retrain'],train['X']); reftest=predict(models['retrain'],test['X'])
            for method,m in models.items():
                ptr=predict(m,train['X']); pte=predict(m,test['X'])
                result={'utility':utility(test['y'],pte),'similarity_F':similarity(ptr[f],reftrain[f]),
                        'similarity_R':similarity(ptr[ri],reftrain[ri]),'similarity_test':similarity(pte,reftest),'cost':costs[method],
                        'mia':{k:mia(train['y'][mi],ptr[mi],test['y'][ni],pte[ni],thresholds) for k,(mi,ni) in cohorts.items()}}
                details[f'{name}/{method}']=result
                rows.append({'fraction':fraction,'method':method,**{k:result['utility'][k] for k in ['macro_f1','pr_auc_ap','fpr']},
                             'prediction_mae_F':result['similarity_F']['prediction_mae'],'prediction_mae_test':result['similarity_test']['prediction_mae'],
                             **{f'mia_auc_{k}':v['mia_auc'] for k,v in result['mia'].items()},**costs[method]})
                checkpoint={'state_dict':state(m),'stage':method,'tensor_sha256':tensor_hash(m),'lineage':lineage[method],
                            'source_manifest_sha256':digest(source/'manifest.json')}
                path=output/'checkpoints'/f'{name}_{method}_final.pt'; torch.save(checkpoint,path)
                reloaded=copy.deepcopy(m); reloaded.load_state_dict(torch.load(path,weights_only=True)['state_dict'])
                if tensor_hash(reloaded)!=tensor_hash(m): raise AssertionError('Final checkpoint mismatch')
                for c in allcoverage:
                    if c['fraction']==fraction and c['method']==method: c['status']='completed'
            if tensor_hash(models['original'])!=tensor_hash(models['no_op']): raise AssertionError('No-op changed')
            # Instrumentation is required to preserve the prior method's exact parameters.
            for method in ['retain_only','fu_no_kd','fu_retain_kd']:
                if tensor_hash(models[method])!=tensor_hash(load(f'{name}_{method}')): raise AssertionError(f'Diagnostic changed {method}')
            write_json(output/'metrics.json',details); pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False)
            print(f'Completed fraction {fraction}; original FU checkpoints reproduced exactly',flush=True)
        write_json(output/'conclusion.json',{'dt_allowed':False,'unlearning_success_claimed':False,
                   'interpretation':'diagnostic pilot; analyze similarity versus retain-only and MIA controls; not a deletion guarantee',
                   'shadow_models_trained':0,'cached_reference_runtime':'see source timing, not a new fair speedup comparison'})
    except Exception:
        (output/'error.log').write_text(traceback.format_exc(),encoding='utf-8'); raise
    finally:
        pd.DataFrame(allcoverage).to_csv(output/'coverage.csv',index=False)
        write_json(output/'timing.json',{'runner_seconds':time.perf_counter()-started,'wire_bytes_measured':False})
        write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source',type=Path,default=Path('runs/pilot_v3_seed42'))
    parser.add_argument('--data',type=Path,default=Path('data/processed/connected_v3_seed42'))
    parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args(); run(a.source,a.data,a.output)
