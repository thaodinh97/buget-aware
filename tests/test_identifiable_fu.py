import unittest
import numpy as np
import torch
from src.data.canary import trigger_dims,stamp,make_canaries
from src.fu.strong_forget import ga_adam,neggrad_plus
from src.fl.fedavg_core import model,ClientWorker

CFG={'lr':1e-2,'batch_size':8,'clip':5.,'ce_margin':3.,'alpha':.9}

class CanaryTests(unittest.TestCase):
    def test_trigger_uses_constant_zero_dims_only(self):
        x=np.array([[1.,0.,0.,2.,0.],[0.,0.,0.,1.,0.]])
        np.testing.assert_array_equal(trigger_dims(x,2),[1,2])
        with self.assertRaises(ValueError): trigger_dims(x,4)

    def test_canaries_are_stamped_relabelled_copies(self):
        x=np.zeros((10,4),np.float32); x[:,0]=np.arange(10); y=np.array([0,1]*5)
        cx,cy,picked=make_canaries(x,y,np.arange(10),3,np.array([2,3]),seed=0)
        self.assertTrue(np.all(y[picked]==1)); np.testing.assert_array_equal(cy,0)
        np.testing.assert_array_equal(cx[:,0],x[picked,0]); self.assertTrue(np.all(cx[:,2:]==1))
        self.assertTrue(np.all(x[:,2:]==0))  # source untouched
        with self.assertRaises(ValueError): make_canaries(x,y,np.arange(10),6,np.array([2]),seed=0)

class StrongForgetTests(unittest.TestCase):
    def setUp(self):
        torch.manual_seed(0); rng=np.random.default_rng(0)
        x=rng.normal(size=(64,4)).astype(np.float32); y=(x[:,0]>0).astype(int)
        self.f=ClientWorker(0,x[:16],y[:16],np.arange(16),device='cpu')
        self.r=ClientWorker(1,x[16:],y[16:],np.arange(16,64),device='cpu')
        self.m=model(4,[8,4],0,device='cpu')

    def ce(self,m,w): return torch.nn.functional.cross_entropy(m(w.x),w.y).item()

    def test_ga_raises_forget_loss_and_leaves_original(self):
        before=[p.clone() for p in self.m.parameters()]
        out,logs=ga_adam(self.m,self.f,CFG,20,seed=1)
        self.assertGreater(self.ce(out,self.f),self.ce(self.m,self.f))
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,self.m.parameters())))
        self.assertTrue(all(r['optimizer_steps']==1 for r in logs if 'stopped' not in r))

    def test_ceiling_stops_ascent(self):
        _,logs=ga_adam(self.m,self.f,{**CFG,'lr':.5,'ce_margin':1.},500,seed=1)
        self.assertEqual(logs[-1]['stopped'],'ce_ceiling'); self.assertLess(len(logs),500)

    def test_ceiling_is_relative_to_starting_forget_loss(self):
        # A poorly fit F (labels flipped: CE far above any absolute cap) must still allow ascent steps.
        flipped=ClientWorker(0,self.f.x.cpu().numpy(),1-self.f.y.cpu().numpy(),np.arange(16),device='cpu')
        _,logs=ga_adam(self.m,flipped,{**CFG,'lr':1e-3},5,seed=1)
        self.assertEqual(sum(r.get('optimizer_steps',0) for r in logs),5)

    def test_neggrad_plus_logs_both_losses(self):
        out,logs=neggrad_plus(self.m,self.f,self.r,CFG,10,seed=1)
        self.assertEqual(len(logs),10); self.assertTrue(all('retain_ce' in r for r in logs))
        # When F and R share one rule, a retain-dominated alpha may lower CE(F); a forget-dominated one must raise it.
        out,_=neggrad_plus(self.m,self.f,self.r,{**CFG,'alpha':.1},10,seed=1)
        self.assertGreater(self.ce(out,self.f),self.ce(self.m,self.f))

    def test_zero_steps_is_a_copy(self):
        out,logs=ga_adam(self.m,self.f,CFG,0,seed=1)
        self.assertEqual(logs,[]); self.assertTrue(all(torch.equal(a,b) for a,b in zip(out.parameters(),self.m.parameters())))

if __name__=='__main__':
    unittest.main()
