"""Read-only local/offline observations, never used for model selection."""
import copy
import hashlib
import time
import numpy as np
import torch
from src.fl.fedavg_core import state, predict
from src.data.prepare import write_json


def tensor_hash(m):
    h=hashlib.sha256()
    for k,v in state(m).items():
        h.update(k.encode()); h.update(str(v.dtype).encode()); h.update(str(tuple(v.shape)).encode())
        h.update(v.numpy().tobytes())
    return h.hexdigest()


def vector(m):
    return torch.cat([p.detach().flatten() for p in m.parameters()])


def loss_gradient(m, worker):
    gradients=[torch.zeros_like(p) for p in m.parameters()]
    total=0.
    for i in range(0,len(worker.y),4096):
        loss=torch.nn.functional.cross_entropy(m(worker.x[i:i+4096]),worker.y[i:i+4096],reduction='sum')/len(worker.y)
        grad=torch.autograd.grad(loss,tuple(m.parameters()))
        total+=loss.item()
        for dest,value in zip(gradients,grad): dest.add_(value.detach())
    return {'ce':total,'full_ce_gradient_l2':sum(g.square().sum() for g in gradients).sqrt().item()}


class Observer:
    def __init__(self, output, prefix, original, forget, retain):
        self.output,self.prefix=output,prefix
        self.workers={'forget':forget,'retain':retain}
        self.origin=vector(original).clone(); self.previous=self.origin.clone()
        self.origin_prediction={k:predict(original,w.x) for k,w in self.workers.items()}
        self.previous_prediction={k:p.copy() for k,p in self.origin_prediction.items()}
        self.parent=tensor_hash(original); self.seconds=0.; self.records=[]

    def __call__(self, stage, m, steps):
        start=time.perf_counter(); was_training=m.training
        m.eval(); before=tensor_hash(m); current=vector(m)
        record={'stage':stage,'optimizer_steps_in_stage':steps,'parent_tensor_sha256':self.parent,
                'tensor_sha256':before,'parameter_l2_from_original':(current-self.origin).norm().item(),
                'parameter_l2_from_previous':(current-self.previous).norm().item(),
                'parameter_max_abs_from_previous':(current-self.previous).abs().max().item()}
        for name,w in self.workers.items():
            p=predict(m,w.x)
            record[name]={**loss_gradient(m,w),'prediction_mae_from_original':float(np.mean(abs(p-self.origin_prediction[name]))),
                          'prediction_mae_from_previous':float(np.mean(abs(p-self.previous_prediction[name])))}
            self.previous_prediction[name]=p
        path=self.output/'checkpoints'/f'{self.prefix}_{stage}.pt'
        torch.save({'state_dict':state(m),'stage':stage,'parent_tensor_sha256':self.parent,'tensor_sha256':before},path)
        restored=copy.deepcopy(m)
        restored.load_state_dict(torch.load(path,weights_only=True)['state_dict'])
        if tensor_hash(restored)!=before or tensor_hash(m)!=before: raise AssertionError('Checkpoint/observer mutated state')
        record['checkpoint']=str(path.name); record['reload_verified']=True
        self.parent=before; self.previous=current.clone()
        m.train(was_training)
        self.seconds+=time.perf_counter()-start
        self.records.append(record)
        write_json(self.output/f'{self.prefix}_diagnostics.json',self.records)
