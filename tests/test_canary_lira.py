import os
import unittest
from unittest import mock
import numpy as np
import pandas as pd
import torch
from src.eval.lira import feature_units,complementary_masks,loo_scores,evaluate_mia
from scripts.run_canary_lira import unit_roles,summarize,train_shadow,MODELS
from scripts.run_canary_placebo import aucs,placebo

class UnitTests(unittest.TestCase):
    def test_identical_rows_share_a_unit(self):
        x=np.array([[1.,0.],[0.,1.],[1.,0.],[1.,-0.]],np.float32)
        unit,sizes=feature_units(x)
        self.assertEqual(unit[0],unit[2]); self.assertEqual(unit[0],unit[3]); self.assertNotEqual(unit[0],unit[1])
        np.testing.assert_array_equal(np.sort(sizes),[1,3])

    def test_units_with_both_cohorts_are_shared(self):
        unit=np.array([0,1,2,1,3,0]); member=np.array([1,1,1,0,0,0],bool)
        np.testing.assert_array_equal(unit_roles(unit,member,4),['shared','shared','member','nonmember'])

class MaskTests(unittest.TestCase):
    def test_complementary_masks_are_balanced(self):
        sizes=np.r_[np.ones(80,int),[5,7,12,30]]
        masks=complementary_masks(sizes,8,67,5,seed=3)
        np.testing.assert_array_equal(masks.sum(0),4)
        np.testing.assert_array_equal(masks[0::2],~masks[1::2])
        self.assertTrue(all(abs(sizes[m].sum()-67)<=5 for m in masks))
        np.testing.assert_array_equal(masks,complementary_masks(sizes,8,67,5,seed=3))
        with self.assertRaises(ValueError): complementary_masks(np.array([100,1]),4,50,5,seed=0,max_draws=50)
        with self.assertRaises(ValueError): complementary_masks(sizes,7,67,5,seed=0)

class ControlTests(unittest.TestCase):
    def test_leave_one_out_detects_planted_membership(self):
        rng=np.random.default_rng(0); n_s,n_u=16,200
        masks=complementary_masks(np.ones(n_u,int),n_s,100,0,seed=1); difficulty=rng.normal(0,3,n_u)
        auc=lambda signals,variant: np.mean([evaluate_mia(m,n,())['auc'] for m,n in loo_scores(signals,masks,variant)])
        planted=difficulty+rng.normal(0,.3,(n_s,n_u))+.5*masks
        self.assertGreater(auc(planted,('online','global')),.75)
        # Per-example calibration removes the spread in difficulty that swamps the raw signal.
        self.assertGreater(auc(planted,('online','global')),auc(planted,None)+.1)
        null=difficulty+rng.normal(0,.3,(n_s,n_u))
        self.assertLess(abs(auc(null,('online','global'))-.5),.06)

class VerdictTests(unittest.TestCase):
    control=pd.DataFrame([{'request':'r','variant':'online_global','control_power':True}])

    def rows(self,shift):
        out=[]
        for s in range(10):
            base=.5+.001*s; noise=.002*(-1)**s
            auc={'original':base+shift,'retrain':base,'retain_only':base+shift,MODELS[3]:base+noise,
                 MODELS[4]:base-noise,MODELS[5]:base+shift/2+noise}
            out.extend({'request':'r','seed':s,'model':k,'variant':'online_global','auc':v,'roc_tpr_at_fpr_0.05':.05} for k,v in auc.items())
        return pd.DataFrame(out)

    def test_verdicts_with_power(self):
        summary,power=summarize(self.rows(.05),self.control); v=summary.set_index('model')
        self.assertTrue(power['r'])
        self.assertEqual(v.loc[MODELS[3],'verdict_vs_retrain'],'RETRAIN_LIKE')
        self.assertEqual(v.loc[MODELS[3],'verdict_vs_original'],'LOWER_THAN_ORIGINAL')
        self.assertEqual(v.loc['retain_only','verdict_vs_retrain'],'DIFFERS_FROM_RETRAIN')
        self.assertEqual(v.loc['retain_only','verdict_vs_original'],'NOT_LOWER')
        self.assertEqual(v.loc['retrain','verdict_vs_retrain'],'RETRAIN_LIKE')
        self.assertAlmostEqual(v.loc['original','leak_remaining'],1.)
        self.assertAlmostEqual(v.loc[MODELS[5],'leak_remaining'],.5)

    def test_without_power_everything_is_inconclusive(self):
        summary,power=summarize(self.rows(0.),self.control)
        self.assertFalse(power['r']); self.assertTrue((summary.verdict_vs_retrain=='MIA_INCONCLUSIVE').all())
        _,power=summarize(self.rows(.05),self.control.assign(control_power=False)); self.assertFalse(power['r'])

class PlaceboTests(unittest.TestCase):
    def test_rank_aucs_match_sklearn_with_ties(self):
        rng=np.random.default_rng(0); scores=np.round(rng.normal(size=(3,40)),1); labels=np.arange(40)<15
        perms=np.stack([labels,rng.permutation(labels)])
        expected=[[evaluate_mia(s[l],s[~l],())['auc'] for s in scores] for l in perms]
        np.testing.assert_allclose(aucs(scores,perms),expected)

    def test_placebo_separates_membership_from_a_fixed_cohort_effect(self):
        rng=np.random.default_rng(1); n=200; labels=np.arange(n)<100; keys=[]; scores=[]
        def models(member_shift):
            for s in range(5):
                difficulty=rng.normal(size=n); retrain=difficulty+.1*rng.normal(size=n)
                # The original ranks rows by a different feature; on a fixed cohort that alone moves the AUC every seed.
                original=difficulty+.5*np.sin(np.arange(n))+member_shift*labels+.1*rng.normal(size=n)
                keys.extend([(s,'retrain'),(s,'original')]); scores.extend([retrain,original])
        models(0.); null_only=placebo(np.array(scores),keys,labels,500,rng)['original']
        self.assertGreater((np.sum(np.abs(null_only[1])>=abs(null_only[0]))+1)/501,.01)
        keys.clear(); scores.clear(); models(1.); planted=placebo(np.array(scores),keys,labels,500,rng)['original']
        self.assertLess((np.sum(np.abs(planted[1])>=abs(planted[0]))+1)/501,.01)

class ShadowTests(unittest.TestCase):
    def test_shadow_appends_pool_rows_to_the_target_client(self):
        rng=np.random.default_rng(0); fl={'rounds':1,'local_epochs':1,'batch_size':8,'lr':1e-3}
        blocks=[(rng.normal(size=(20,4)).astype(np.float32),rng.integers(0,2,20),np.array([f'r{j}_{i}' for i in range(20)])) for j in range(3)]
        x=rng.normal(size=(7,4)).astype(np.float32)
        with mock.patch.dict(os.environ,{'FL_DEVICE':'cpu'}):
            m,logs=train_shadow(blocks,1,x,5,[8,4],fl); again,_=train_shadow(blocks,1,x,5,[8,4],fl)
        self.assertEqual(logs[0]['counts'],[20,27,20])
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(m.state_dict().values(),again.state_dict().values())))

if __name__=='__main__':
    unittest.main()
