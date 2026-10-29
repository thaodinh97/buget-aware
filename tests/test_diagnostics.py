import copy
import tempfile
import unittest
from pathlib import Path
import numpy as np
import torch
from src.fl.fedavg_core import model,ClientWorker
from src.fu.conpfu_inspired import alternating,distill
from src.fu.diagnostics import Observer,tensor_hash,loss_gradient
from src.eval.metrics import mia,calibrate_mia
from scripts.run_controlled_diagnostics import family_match


class DiagnosticTests(unittest.TestCase):
    def setUp(self):
        torch.set_num_threads(1)
        self.m=model(2,[4,3],42)
        self.w=ClientWorker(0,np.array([[1.,0.],[0.,1.],[1.,1.],[.5,.2]]),np.array([0,1,1,0]),['a','b','c','d'])
        self.cfg={'cycles':2,'batch_size':2,'forget_lr':.001,'retain_lr':.001,'regularization':.001,'clip':1.,
                  'kd_epochs':1,'kd_lr':.001,'temperature':2.,'hard_weight':.5}

    def test_observer_preserves_updates_and_step_counts(self):
        plain,_=alternating(self.m,self.w,self.w,self.cfg,42)
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            out=Path(tmp); (out/'checkpoints').mkdir()
            obs=Observer(out,'test',self.m,self.w,self.w)
            observed,logs=alternating(self.m,self.w,self.w,self.cfg,42,observer=obs)
            self.assertEqual(tensor_hash(plain),tensor_hash(observed))
            self.assertEqual(sum(l['optimizer_steps'] for l in logs),8)
            self.assertEqual(sum(r['optimizer_steps_in_stage'] for r in obs.records),8)
            self.assertEqual(len(obs.records),8)
            self.assertTrue(all(r['reload_verified'] for r in obs.records))
            self.assertTrue(all(l['gradient_norm_after_clip']<=1.00001 for l in logs))
            self.assertTrue(all(l['parameter_step_l2']>0 for l in logs))

    def test_full_gradient_and_observation_do_not_change_params(self):
        before=tensor_hash(self.m)
        result=loss_gradient(self.m,self.w)
        loss=torch.nn.functional.cross_entropy(self.m(self.w.x),self.w.y)
        grads=torch.autograd.grad(loss,tuple(self.m.parameters()))
        norm=sum(g.square().sum() for g in grads).sqrt().item()
        self.assertAlmostEqual(result['ce'],loss.item(),places=6)
        self.assertAlmostEqual(result['full_ce_gradient_l2'],norm,places=6)
        self.assertEqual(tensor_hash(self.m),before)

    def test_kd_observation_preserves_teacher_and_updates(self):
        teacher,_=alternating(self.m,self.w,self.w,self.cfg,42)
        before=tensor_hash(teacher)
        plain,_=distill(self.m,teacher,self.w,self.cfg,42)
        with tempfile.TemporaryDirectory(dir='.') as tmp:
            out=Path(tmp); (out/'checkpoints').mkdir()
            obs=Observer(out,'kd',self.m,self.w,self.w)
            measured,_=distill(self.m,teacher,self.w,self.cfg,42,observer=obs)
            self.assertEqual(tensor_hash(plain),tensor_hash(measured))
            self.assertEqual(tensor_hash(teacher),before)
            self.assertEqual(obs.records[0]['tensor_sha256'],tensor_hash(self.m))

    def test_attack_null_and_inverted_direction_not_flipped(self):
        y=np.array([0,1,0,1]); p=np.array([.2,.8,.2,.8])
        t=calibrate_mia(y,p)
        self.assertEqual(mia(y,p,y,p,t)['mia_auc'],.5)
        self.assertEqual(mia(y,np.array([.4,.6,.4,.6]),y,p,t)['mia_auc'],0.)

    def test_family_matching_exact_and_unsupported_recorded(self):
        train={'y':np.array([0,1,1,1])}; test={'y':np.array([0,1])}
        types={'train':np.array(['normal','a','a','missing']),'test':np.array(['normal','a'])}
        mi,ni,coverage=family_match(np.arange(4),train,test,types,42)
        np.testing.assert_array_equal(train['y'][mi],test['y'][ni])
        np.testing.assert_array_equal(types['train'][mi],types['test'][ni])
        self.assertEqual(next(c for c in coverage if c['type']=='missing')['selected_each'],0)


if __name__=='__main__': unittest.main()
