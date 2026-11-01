"""Read and verify existing experiment inputs without changing artifacts."""
import json
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from scripts.run_pilot import load_data
from src.data.prepare import digest,write_json
from src.fl.fedavg_core import model,get_device

CASES=['client0_random_stratified_0.1','client4_random_stratified_0.5']

class Context:
    def __init__(self,source=Path('runs/controlled_phase_v2_seed42'),data_path=Path('data/processed/connected_v3_seed42')):
        self.source=source; self.data_path=data_path
        for name,sha in json.loads((source/'artifact_hashes.json').read_text()).items():
            if digest(source/name)!=sha: raise ValueError(f'Changed source artifact: {name}')
        self.manifest=json.loads((source/'manifest.json').read_text()); self.cfg=self.manifest['base_config']
        torch.set_num_threads(self.cfg['threads']); torch.use_deterministic_algorithms(True)
        self.data=load_data(data_path); self.tr=self.data['train']
        if digest(data_path/'manifest.json')!=self.manifest['data_manifest_sha256']:
            raise ValueError('Wrong frozen dataset')
        self.metadata=pd.read_csv(data_path/'metadata.csv',dtype=str).set_index('row_id')
        self.types={s:self.metadata.loc[d['row_id'],'type'].to_numpy() for s,d in self.data.items()}
        self.index={v:i for i,v in enumerate(self.tr['row_id'])}
        source_parts=Path(self.manifest['source'])/'partitions.npz'
        source_hashes=json.loads((source_parts.parent/'artifact_hashes.json').read_text())
        if digest(source_parts)!=source_hashes['partitions.npz']:
            raise ValueError('Changed frozen client partitions')
        if digest(source_parts.parent/'manifest.json')!=self.manifest['source_manifest_sha256']:
            raise ValueError('Changed original manifest')
        with np.load(source_parts) as z:
            self.parts=[np.array([self.index[v] for v in z[f'client_{i}']]) for i in range(5)]
        self.selections=json.loads((source/'selections.json').read_text())

    def load(self,case,name):
        m=model(self.tr['X'].shape[1],self.cfg['hidden'],self.cfg['seed'])
        ck=torch.load(self.source/'checkpoints'/f'{case}_{name}.pt',weights_only=True)
        m.load_state_dict(ck['state_dict']); return m

    def request(self,case):
        request=json.loads((self.source/'requests'/f'{case}.json').read_text())
        f=np.array([self.index[v] for v in request['forget_ids']])
        retained=[np.setdiff1d(p,f) for p in self.parts]
        return f,retained,request

    def manifest_for(self,output,kind,protocol):
        output.mkdir(parents=True,exist_ok=False)
        write_json(output/'manifest.json',{'kind':kind,'source':str(self.source),
            'source_manifest_sha256':digest(self.source/'manifest.json'),
            'data_manifest_sha256':digest(self.data_path/'manifest.json'),'config':self.cfg,'protocol':protocol,
            'device':str(get_device()),
            'sources':{str(p):digest(p) for root in ['src','scripts'] for p in Path(root).rglob('*.py')},
            'torch':torch.__version__,'numpy':np.__version__,'split_regenerated':False,'target_seed_count':1})

def use_run_device(run):
    """Re-execute on the device that produced a run: CPU and CUDA kernels differ in the last bits,
    so bit-for-bit re-execution checks only hold on the recorded device."""
    device=json.loads((Path(run)/'manifest.json').read_text()).get('device')
    if device is None: raise ValueError(f'{run} records no device; re-execution would be ambiguous')
    os.environ['FL_DEVICE']=device
    return device

def finalize(output):
    write_json(output/'artifact_hashes.json',{str(p.relative_to(output)):digest(p)
               for p in output.rglob('*') if p.is_file() and p.name!='artifact_hashes.json'})
