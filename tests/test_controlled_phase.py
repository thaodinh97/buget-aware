import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.fl.fedavg_core import model,ClientWorker
from src.fu.step_budget import scheduled,full_batches
from src.fu.diagnostics import Observer,tensor_hash
from src.data.request_sensitivity import build_request
from src.eval.shadow_mia import auxiliary_buckets,assess_scores
from scripts.run_controlled_phase import select_ratio
from src.profiling import CostProfile

class ControlledTests(unittest.TestCase):
    def test_profile_round_roles(self):
        with CostProfile() as prof:
            prof.observe(lambda: None)
        result=prof.result(steps=5,exposures=10,training_rounds=3)
        self.assertEqual(result['recovery_rounds'],0)
        self.assertEqual(result['training_rounds'],3)
        self.assertEqual(result['optimizer_steps'],5)
        self.assertGreater(result['rss_samples'],0)
        self.assertGreater(result['rss_sampled_max_bytes'],0)
        self.assertFalse(result['wire_bytes_measured'])

    def test_steps_and_observer(self):
        torch.set_num_threads(1)
        m=model(2,[4,3],42)
        w=ClientWorker(0,np.eye(2,dtype=np.float32),np.array([0,1]),['a','b'])
        cfg=dict(batch_size=8,forget_lr=.001,retain_lr=.001,regularization=.001,clip=1.)
        plan=[('ga',3),('gd',1)]*2
        plain,logs=scheduled(m,w,w,cfg,plan,42)
        self.assertEqual(len(logs),8)
        self.assertEqual(sum(l['samples'] for l in logs),64)
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            out=Path(tmp); (out/'checkpoints').mkdir()
            obs=Observer(out,'check',m,w,w)
            observed,_=scheduled(m,w,w,cfg,plan,42,obs)
            self.assertEqual(tensor_hash(plain),tensor_hash(observed))
            self.assertTrue(all(r['reload_verified'] for r in obs.records))
        noop,logs=scheduled(m,w,w,cfg,[],42)
        self.assertEqual(tensor_hash(m),tensor_hash(noop)); self.assertEqual(logs,[])

    def test_batches_and_group_requests(self):
        a=full_batches(3,8,42); b=full_batches(3,8,42)
        for _ in range(4): np.testing.assert_array_equal(next(a),next(b))
        tr={'y':np.tile([0,1],20),'group_id':np.repeat(np.arange(20),2)}
        parts=[np.arange(32),np.arange(32,40)]; sig=np.repeat(np.arange(20),2)
        for frac in [.25,.5]:
            f,r,audit=build_request(tr,parts,0,frac,'low_overlap_groups',sig,42)
            self.assertFalse(set(tr['group_id'][f]) & set(tr['group_id'][r]))
            self.assertEqual(audit['feature_equivalent_overlap_fraction'],0)
            self.assertEqual(len(f)+len(r),32)

    def test_auxiliary_disjoint_and_attack_direction(self):
        groups=np.repeat(np.arange(80),3)
        buckets=auxiliary_buckets(groups,42)
        self.assertEqual(sum(map(len,buckets)),len(groups))
        for i in range(8):
            for j in range(i): self.assertFalse(set(groups[buckets[i]]) & set(groups[buckets[j]]))
        self.assertEqual(assess_scores(np.array([.9,.8]),np.array([.1,.2]),.5)['auc'],1)
        self.assertEqual(assess_scores(np.array([.1,.2]),np.array([.9,.8]),.5)['auc'],0)

    def test_selection_constraints(self):
        cfg={'validation_max_f1_drop':.03,'validation_max_fpr_increase':.03}
        original={'macro_f1':.9,'fpr':.1}
        candidates={'alt_3_1':dict(macro_f1=.8,fpr=.1,prediction_mae=0),
                    'alt_1_1':dict(macro_f1=.9,fpr=.1,prediction_mae=.2)}
        self.assertEqual(select_ratio(candidates,original,cfg)[0],'alt_1_1')
        self.assertEqual(select_ratio({},original,cfg)[1],'NO_FEASIBLE_VALIDATION_ARM')
