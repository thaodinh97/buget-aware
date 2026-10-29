"""Small validation-selected FU pilot on frozen data, with independent learned MIA."""
import argparse
import copy
import json
import platform
import sys
import time
import traceback
from datetime import datetime, timezone
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import numpy as np
import pandas as pd
import torch
from scripts.run_pilot import load_data
from scripts.verify_run import verify
from scripts.run_controlled_diagnostics import family_match
from src.data.prepare import digest,write_json
from src.data.federated import ids_hash
from src.data.request_sensitivity import build_request
from src.fl.fedavg_core import model,state,fedavg,ClientWorker,predict,get_device
from src.fu.diagnostics import Observer,tensor_hash
from src.fu.step_budget import scheduled,kd_steps
from src.eval.metrics import utility,similarity
from src.eval.shadow_mia import train_attack
from src.profiling import CostProfile


def select_ratio(validation,original,cfg):
    eligible=[name for name,v in validation.items() if name.startswith('alt_')
              and v['macro_f1']>=original['macro_f1']-cfg['validation_max_f1_drop']
              and v['fpr']<=original['fpr']+cfg['validation_max_fpr_increase']]
    if not eligible: return 'alt_1_1','NO_FEASIBLE_VALIDATION_ARM'
    return min(eligible,key=lambda name:(validation[name]['prediction_mae'],name)),'selected_on_validation'


def measured_fl(initial,workers,cfg,rounds,seed,training=False):
    with CostProfile() as prof:
        result,logs=fedavg(initial,workers,cfg,rounds,seed)
    return result,logs,prof.result(steps=sum(c['steps'] for e in logs for c in e['clients']),
              exposures=sum(len(w.y)*cfg['local_epochs'] for w in workers)*rounds,
              rounds=0 if training else rounds,training_rounds=rounds if training else 0,
              up=sum(e['payload_bytes_up'] for e in logs),down=sum(e['payload_bytes_down'] for e in logs))


def combine_cost(parts):
    return {**{key:sum(p[key] for p in parts) for key in ['wall_seconds','process_cpu_seconds','optimizer_steps',
            'sample_exposures','recovery_rounds','training_rounds','payload_bytes_up','payload_bytes_down']},
            'rss_sampled_max_bytes':max((p['rss_sampled_max_bytes'] for p in parts),default=0)}


def run(source,data_path,config_path,output):
    if output.exists(): raise FileExistsError(output)
    source_verification=verify(source,data_path,allow_changed_reports=True)
    src=json.loads((source/'manifest.json').read_text()); base=src['config']
    cfg=json.loads(config_path.read_text()); data=load_data(data_path)
    torch.set_num_threads(base['threads']); torch.use_deterministic_algorithms(True)
    device=get_device()
    print(f'Using device: {device}',flush=True)
    tr,va,te=data['train'],data['validation'],data['test']
    metadata=pd.read_csv(data_path/'metadata.csv',dtype=str).set_index('row_id')
    types={s:metadata.loc[d['row_id'],'type'].to_numpy() for s,d in data.items()}
    signatures=metadata.loc[tr['row_id'],'feature_signature'].to_numpy()
    mapping={rid:i for i,rid in enumerate(tr['row_id'])}
    with np.load(source/'partitions.npz',allow_pickle=False) as z:
        parts=[np.array([mapping[r] for r in z[f'client_{i}']]) for i in range(5)]
    original=model(tr['X'].shape[1],base['hidden'],base['seed'])
    ckpt=torch.load(source/'checkpoints/original.pt',weights_only=True)
    if ckpt['manifest_sha256']!=digest(source/'manifest.json'): raise ValueError('Wrong original checkpoint')
    original.load_state_dict(ckpt['state_dict'])
    output.mkdir(parents=True); (output/'checkpoints').mkdir(); (output/'requests').mkdir()
    manifest={'source':str(source),'source_verification':source_verification,'source_manifest_sha256':digest(source/'manifest.json'),
              'data_manifest_sha256':digest(data_path/'manifest.json'),'base_config':base,'phase_config':cfg,
              'torch':torch.__version__,'python':platform.python_version(),'hardware':platform.processor(),
              'device':str(device),
              'sources':{str(p):digest(p) for root in ['src','scripts'] for p in Path(root).rglob('*.py')},
              'split_regenerated':False,'selection':'validation only; choices frozen before each request test evaluation',
              'scope':'single seed pilot; process profiling, simulated payload; no planner',
              'algorithm_attribution':'ConPFU-inspired global-start step-budget/full-local-KD adaptation, not faithful reproduction'}
    write_json(output/'manifest.json',manifest)
    started=time.perf_counter(); coverage=[]; requests={}; rows=[]; evals={}; selections={}; failures=[]
    for target in cfg['targets']:
        for design in cfg['request_designs']:
            for fraction in cfg['fractions']:
                key=f'client{target}_{design}_{fraction}'
                coverage.append({'request':key,'status':'planned'})
                try:
                    f,r,audit=build_request(tr,parts,target,fraction,design,signatures,cfg['seed']+1)
                    if target==4 and design=='random_stratified':
                        old=json.loads((source/f'request_{fraction}.json').read_text())
                        if ids_hash(tr['row_id'][f])!=old['forget_hash']: raise AssertionError('Prior random request changed')
                    audit.update(forget_ids=tr['row_id'][f].tolist(),retain_ids=tr['row_id'][r].tolist(),
                         forget_hash=ids_hash(tr['row_id'][f]),forget_types=pd.Series(types['train'][f]).value_counts().to_dict(),
                         retain_types=pd.Series(types['train'][r]).value_counts().to_dict())
                    write_json(output/'requests'/f'{key}.json',audit)
                    requests[key]=(f,r,target,audit)
                except ValueError as exc:
                    coverage[-1]['status']='invalid_request'; failures.append({'request':key,'error':str(exc)})
    write_json(output/'request_inventory.json',{'requests':{p.name:digest(p) for p in (output/'requests').iterdir()},
                                               'coverage':coverage,'failures':failures})
    def worker(i,indices): return ClientWorker(i,tr['X'][indices],tr['y'][indices],tr['row_id'][indices])
    size=sum(v.numel()*v.element_size() for v in state(original).values())
    def save(key,name,m,parent=None,teacher=None):
        path=output/'checkpoints'/f'{key}_{name}.pt'
        torch.save({'state_dict':state(m),'stage':name,'parent_tensor_sha256':parent,'teacher_tensor_sha256':teacher,
                    'tensor_sha256':tensor_hash(m),'manifest_sha256':digest(output/'manifest.json')},path)
        loaded=copy.deepcopy(m); loaded.load_state_dict(torch.load(path,weights_only=True)['state_dict'])
        if tensor_hash(loaded)!=tensor_hash(m): raise AssertionError('Checkpoint reload mismatch')
    try:
        attack=train_attack(data['calibration'],types['calibration'],original,base,cfg,output/'attack')
        original_val=utility(va['y'],predict(original,va['X']))
        for key,(f,r,target,audit) in requests.items():
            print(f'Start {key}: {len(f)} forget samples',flush=True)
            record=next(c for c in coverage if c['request']==key); record['status']='running'
            case=output/key; case.mkdir(); (case/'checkpoints').mkdir()
            profiles={}; diagnostics={}; candidates={}; validation={}; models={}; costs={}; parents={}; teachers={}
            try:
                retained=[np.setdiff1d(p,f) for p in parts]; ri=np.concatenate(retained)
                if np.intersect1d(ri,f).size: raise AssertionError('Forgotten samples in recovery')
                workers=[worker(i,idx) for i,idx in enumerate(retained)]; fw,rw=worker(target,f),worker(target,r)
                initial=model(tr['X'].shape[1],base['hidden'],base['seed'])
                fresh=copy.deepcopy(initial); source_initial=torch.load(source/'checkpoints/initial.pt',weights_only=True)
                if any(not torch.equal(v,source_initial['state_dict'][k]) for k,v in state(fresh).items()):
                    raise AssertionError('Fresh initialization policy changed')
                retrain,logs,cp=measured_fl(fresh,workers,base['fl'],base['fl']['rounds'],base['seed'],training=True)
                profiles['retrain']=cp; write_json(case/'retrain_training.json',logs)
                refval=predict(retrain,va['X'])
                models.update(original=original,no_op=copy.deepcopy(original),retrain=retrain)
                parents.update(original=None,no_op=tensor_hash(original),retrain=tensor_hash(fresh))
                costs.update(original=[],no_op=[],retrain=['retrain'])
                for arm,schedule in cfg['schedules'].items():
                    obs=Observer(case,arm,original,fw,rw)
                    with CostProfile() as prof:
                        pre,step_logs=scheduled(original,fw,rw,base['fu'],schedule,base['seed'],
                                              observer=lambda *args:prof.observe(obs,*args))
                    count=sum(s['optimizer_steps'] for s in step_logs)
                    profiles[f'{arm}.local']=prof.result(steps=count,exposures=count*base['fu']['batch_size'],
                                                       up=size if count else 0,down=size if count else 0)
                    diagnostics[arm]=obs.seconds; write_json(case/f'{arm}_steps.json',step_logs)
                    post,logs,cp=measured_fl(pre,workers,base['fl'],base['recovery_rounds'],base['seed']+900000)
                    profiles[f'{arm}.recovery']=cp; write_json(case/f'{arm}_recovery.json',logs)
                    candidates[arm]=(pre,post)
                    for stage,m in [('pre',pre),('post',post)]:
                        p=predict(m,va['X']); u=utility(va['y'],p)
                        validation[f'{arm}/{stage}']={**u,**similarity(p,refval)}
                        models[f'{arm}/{stage}']=m
                        parents[f'{arm}/{stage}']=tensor_hash(original if stage=='pre' else pre)
                        costs[f'{arm}/{stage}']=[f'{arm}.local']+([f'{arm}.recovery'] if stage=='post' else [])
                chosen,status=select_ratio({arm:validation[f'{arm}/post'] for arm in cfg['schedules']},original_val,cfg)
                selections[key]={'chosen':chosen,'status':status,'validation_candidates':validation,
                                 'selection_timestamp_utc':datetime.now(timezone.utc).isoformat(),
                                 'rule':'post recovery validation MAE to retrain under utility constraints; no test inputs'}
                write_json(case/'selection.json',selections[key]); write_json(output/'selections.json',selections)
                for arm,teacher,kd_worker,parent in [
                    ('ga_gd_with_kd',candidates[chosen][0],rw,chosen),
                    ('conpfu_inspired',candidates[chosen][0],worker(target,parts[target]),chosen),
                    ('retain_kd_matched',candidates['gd_only'][0],rw,'gd_only')]:
                    obs=Observer(case,arm,original,fw,rw)
                    with CostProfile() as prof:
                        pre,step_logs=kd_steps(original,teacher,kd_worker,base['fu'],cfg['kd_steps'],base['seed'],
                                              observer=lambda *args:prof.observe(obs,*args))
                    profiles[f'{arm}.kd']=prof.result(steps=len(step_logs),exposures=len(step_logs)*base['fu']['batch_size'])
                    diagnostics[arm]=obs.seconds; write_json(case/f'{arm}_steps.json',step_logs)
                    post,logs,cp=measured_fl(pre,workers,base['fl'],base['recovery_rounds'],base['seed']+900000)
                    profiles[f'{arm}.recovery']=cp; write_json(case/f'{arm}_recovery.json',logs)
                    for stage,m in [('pre',pre),('post',post)]:
                        models[f'{arm}/{stage}']=m
                        parents[f'{arm}/{stage}']=tensor_hash(original if stage=='pre' else pre)
                        teachers[f'{arm}/{stage}']=tensor_hash(teacher)
                        costs[f'{arm}/{stage}']=[f'{parent}.local',f'{arm}.kd']+([f'{arm}.recovery'] if stage=='post' else [])
                write_json(case/'profiles.json',profiles); write_json(case/'diagnostic_seconds.json',diagnostics)
                # All decisions are now frozen. Test is only used below for fixed reporting.
                cohorts={}; mia_cov={}
                pools={'F':f,'R_all':ri,'R_group':ri[~np.isin(tr['group_id'][ri],tr['group_id'][f])]}
                for label,pool in pools.items():
                    try:
                        mi,ni,cov=family_match(pool,tr,te,types,base['seed'])
                        cohorts[label]=(mi,ni); mia_cov[label]=cov
                    except ValueError as exc: mia_cov[label]={'status':'inconclusive_no_support','error':str(exc)}
                write_json(case/'mia_coverage.json',mia_cov)
                np.savez_compressed(case/'mia_ids.npz',**{f'{k}_{side}':(tr if side=='member' else te)['row_id'][idx]
                           for k,pair in cohorts.items() for side,idx in zip(['member','nonmember'],pair)})
                ptr_ref=predict(retrain,tr['X']); pte_ref=predict(retrain,te['X'])
                for name,m in models.items():
                    ptr=predict(m,tr['X']); pte=predict(m,te['X'])
                    result={'utility':utility(te['y'],pte),'similarity_F':similarity(ptr[f],ptr_ref[f]),
                            'similarity_R':similarity(ptr[ri],ptr_ref[ri]),'similarity_test':similarity(pte,pte_ref),
                            'mia':{c:attack.assess(tr['y'][mi],ptr[mi],te['y'][ni],pte[ni]) for c,(mi,ni) in cohorts.items()},
                            'cost':combine_cost([profiles[p] for p in costs[name]]),'cost_parts':costs[name]}
                    evals[f'{key}/{name}']=result
                    role='ablation'
                    if name in ['original','no_op','retrain']: role=name
                    elif name=='no_update/post': role='retain_only_recovery'
                    elif name==f'{chosen}/post': role='ga_gd_without_kd'
                    elif name=='gd_only/post': role='compute_matched_retained'
                    elif name.endswith('/post'): role=name.split('/')[0]
                    rows.append({'request':key,'target':target,'design':audit['design'],'fraction':audit['requested_fraction'],
                       'actual_fraction':audit['actual_fraction'],'method':name,'comparison_role':role,'cached_reference':name in ['original','no_op'],
                       'overlap_fraction':audit['feature_equivalent_overlap_fraction'],
                       **{k:result['utility'][k] for k in ['macro_f1','pr_auc_ap','fpr']},
                       'prediction_mae_F':result['similarity_F']['prediction_mae'],
                       'prediction_mae_test':result['similarity_test']['prediction_mae'],
                       **{f'mia_{metric}_{c}':values[metric] for c,values in result['mia'].items()
                          for metric in ['auc','tpr_at_calibrated_fpr_0_01','realized_nonmember_fpr']},**result['cost']})
                    save(key,name.replace('/','_'),m,parents[name],teachers.get(name))
                if tensor_hash(models['no_op'])!=tensor_hash(original) or tensor_hash(candidates['no_update'][0])!=tensor_hash(original):
                    raise AssertionError('No-update control changed')
                controls=evals[f'{key}/original']['mia']
                attack_status='inconclusive' if attack.validation['auc']<.55 or 'R_group' not in controls or controls['R_group']['auc']<.55 else 'controls_detectable_not_deletion_proof'
                write_json(case/'conclusion.json',{'attack_status':attack_status,'dt_allowed':False,'unlearning_success_claimed':False,
                     'selection':status,'caution':'one seed; auxiliary shadow transfer and grouping shifts; sample/step matching not exact FLOPs'})
                record['status']='completed'; record['evaluated_entries']=len(models)
                print(f'Completed {key}; selected {chosen}; MIA {attack_status}',flush=True)
            except Exception as exc:
                record['status']='failed'; failures.append({'request':key,'error':str(exc)})
                (case/'error.log').write_text(traceback.format_exc(),encoding='utf-8')
            finally:
                pd.DataFrame(rows).to_csv(output/'metrics.csv',index=False); write_json(output/'metrics.json',evals)
                pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False); write_json(output/'failures.json',failures)
        write_json(output/'conclusion.json',{'dt_allowed':False,'unlearning_success_claimed':False,
                    'completed_requests':sum(c['status']=='completed' for c in coverage),'planned_requests':8,
                    'shadow_models_trained':3,'seed_count':1})
    finally:
        write_json(output/'runtime.json',{'total_runner_seconds':time.perf_counter()-started,'includes_diagnostics':True})
        pd.DataFrame(coverage).to_csv(output/'coverage.csv',index=False)
        write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):digest(p) for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--source',type=Path,default=Path('runs/pilot_v3_seed42'))
    p.add_argument('--data',type=Path,default=Path('data/processed/connected_v3_seed42'))
    p.add_argument('--config',type=Path,default=Path('configs/controlled_phase.json'))
    p.add_argument('--output',type=Path,required=True)
    a=p.parse_args(); run(a.source,a.data,a.config,a.output)
