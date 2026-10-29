import unittest
import numpy as np
import torch
from scipy.stats import norm
from src.eval.lira import LiRA,lira_signal,balanced_masks,evaluate_mia
from src.eval.membership_validation import logit_margin

class Fixed(torch.nn.Module):
    def __init__(self,logits):
        super().__init__(); self.logits=torch.nn.Parameter(torch.as_tensor(logits,dtype=torch.float32))
    def forward(self,x): return self.logits[x[:,0].long()]

class LiRATests(unittest.TestCase):
    def test_signal_is_finite_signed_margin_at_saturation(self):
        m=Fixed([[0.,120.],[0.,-60.],[3.,1.]]); x=np.arange(3,dtype=np.float32)[:,None]; y=np.array([1,1,0])
        s=lira_signal(m,x,y)
        self.assertTrue(np.all(np.isfinite(s)))
        np.testing.assert_allclose(s,(2*y-1)*logit_margin(m,x))
        np.testing.assert_allclose(s,[120.,-60.,2.],atol=1e-5)

    def test_multiclass_signal_is_logit_of_true_class_probability(self):
        z=np.array([[1.,2.,.5],[-1.,0.,3.]]); y=np.array([1,2])
        s=lira_signal(Fixed(z),np.arange(2,dtype=np.float32)[:,None],y)
        p=np.exp(z)/np.exp(z).sum(1,keepdims=True); py=p[np.arange(2),y]
        np.testing.assert_allclose(s,np.log(py/(1-py)),rtol=1e-6)

    def test_balanced_masks_share_units(self):
        units=np.array([0,0,1,2,2,2,3])
        masks=balanced_masks(8,units,seed=1)
        self.assertEqual(masks.shape,(8,7)); np.testing.assert_array_equal(masks.sum(0),4)
        np.testing.assert_array_equal(masks[:,0],masks[:,1]); np.testing.assert_array_equal(masks[:,3],masks[:,5])
        with self.assertRaises(ValueError): balanced_masks(7,units,seed=1)

    def test_online_and_offline_match_analytic_gaussians(self):
        masks=np.array([[1,0],[1,1],[1,0],[0,1],[0,0],[0,1]],bool)
        signals=np.array([[3.,0.],[5.,3.],[4.,1.],[1.,4.],[0.,2.],[2.,5.]])
        on=LiRA('online','per_example').fit(signals,masks)
        np.testing.assert_allclose(on.in_mean,[4.,4.]); np.testing.assert_allclose(on.out_mean,[1.,1.])
        sd=np.sqrt(2/3)
        expected=norm.logpdf(3.5,4,sd)-norm.logpdf(3.5,1,sd)
        np.testing.assert_allclose(on.score([3.5],[0]),[expected])
        glob=LiRA('online','global').fit(signals,masks)
        np.testing.assert_allclose(glob.in_std,[np.sqrt(4/6)]*2); np.testing.assert_allclose(glob.out_std,[np.sqrt(4/6)]*2)
        off=LiRA('offline','per_example').fit(signals,masks)
        scores=off.score([0.,1.,2.,8.],[0,0,0,0])
        self.assertTrue(np.all(np.diff(scores)>0))
        np.testing.assert_allclose(scores[1],np.log(.5))

    def test_group_support_requirements(self):
        signals=np.zeros((4,2)); masks=np.array([[1,0],[0,0],[0,0],[0,0]],bool)
        with self.assertRaises(ValueError): LiRA('online').fit(signals,masks)
        LiRA('offline').fit(signals,np.zeros((4,2),bool))
        with self.assertRaises(ValueError): LiRA('online').fit(np.full((4,2),np.inf),masks)

    def test_evaluate_mia_low_fpr_operating_point(self):
        r=evaluate_mia([5,4,3,0],[2,1,0,-1])
        self.assertEqual(r['auc'],(4+4+4+1.5)/16); self.assertEqual(r['roc_tpr_at_fpr_0.001'],.75)
        same=evaluate_mia([1,2,3],[1,2,3]); self.assertEqual(same['auc'],.5)

    def test_toy_memorization_is_detected_leave_one_shadow_out(self):
        torch.manual_seed(0); rng=np.random.default_rng(0)
        x=rng.normal(size=(40,5)).astype(np.float32); y=rng.integers(0,2,40)
        masks=balanced_masks(16,np.arange(40),seed=2); signals=[]
        for k in range(16):
            torch.manual_seed(k); m=torch.nn.Sequential(torch.nn.Linear(5,64),torch.nn.ReLU(),torch.nn.Linear(64,2))
            opt=torch.optim.Adam(m.parameters(),lr=.02); idx=np.flatnonzero(masks[k])
            xb,yb=torch.as_tensor(x[idx]),torch.as_tensor(y[idx])
            for _ in range(200):
                opt.zero_grad(); torch.nn.functional.cross_entropy(m(xb),yb).backward(); opt.step()
            signals.append(lira_signal(m,x,y))
        signals=np.array(signals); aucs=[]
        for held in range(4):
            keep=np.arange(16)!=held
            attack=LiRA('online','global').fit(signals[keep],masks[keep])
            s=attack.score(signals[held]); aucs.append(evaluate_mia(s[masks[held]],s[~masks[held]])['auc'])
        self.assertGreater(np.mean(aucs),.8)

if __name__=='__main__':
    unittest.main()
