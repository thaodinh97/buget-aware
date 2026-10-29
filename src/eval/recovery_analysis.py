"""Offline geometry/calibration measurements; never used in training decisions."""
import numpy as np
import torch

def vector(state):
    return torch.cat([v.detach().reshape(-1).double() for v in state.values()])

def alignment(before,after,reference):
    delta=after-before; direction=reference-before
    dn=delta.norm().item(); rn=direction.norm().item()
    dot=torch.dot(delta,direction).item()
    return {'update_norm':dn,'direction_norm':rn,
            'cosine_to_retrain':dot/(dn*rn) if dn>0 and rn>0 else None,
            'zero_update':dn==0,'zero_reference_direction':rn==0,
            'distance_before':rn,'distance_after':(reference-after).norm().item(),
            'squared_distance_change':dn*dn-2*dot,
            'dot_to_retrain':dot}

def calibration(y,p):
    bins=np.minimum((p*10).astype(int),9); ece=0.
    for b in range(10):
        mask=bins==b
        if mask.any(): ece+=mask.mean()*abs(y[mask].mean()-p[mask].mean())
    return {'brier':float(np.mean((y-p)**2)),'ece_10_equal_width':float(ece),
            'mean_confidence':float(np.maximum(p,1-p).mean())}
