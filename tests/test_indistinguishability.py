import unittest
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from src.eval.indistinguishability import loo_lr_scores,pair_auc,distinguish,swap_null
from src.eval.lira import LiRA
from scripts.run_indistinguishability import pair_kinds,verdicts,observations,COMPARATORS

class StatisticTests(unittest.TestCase):
    def test_loo_scores_equal_lira_with_global_variance(self):
        rng=np.random.default_rng(0); c=rng.normal(size=(6,15)); r=rng.normal(.2,1.3,size=(6,15))
        sc,sr=loo_lr_scores(c,r)
        for s in range(6):
            keep=np.arange(6)!=s; attack=LiRA('online','global').fit(np.r_[c[keep],r[keep]],np.r_[np.ones((5,15),bool),np.zeros((5,15),bool)])
            self.assertAlmostEqual(sc[s],attack.score(c[s]).sum(),places=9); self.assertAlmostEqual(sr[s],attack.score(r[s]).sum(),places=9)

    def test_pair_auc_matches_sklearn_with_ties(self):
        rng=np.random.default_rng(1); a=np.round(rng.normal(size=30),1); b=np.round(rng.normal(.3,size=25),1)
        self.assertAlmostEqual(pair_auc(a,b),roc_auc_score(np.r_[np.ones(30),np.zeros(25)],np.r_[a,b]))

    def test_planted_difference_is_detected_and_null_is_not(self):
        rng=np.random.default_rng(2); n,u=20,50; r=rng.normal(size=(n,u))
        shifted=rng.normal(size=(n,u))+1.*(np.arange(u)<10)
        res,_,_=distinguish(shifted,r,200,200,np.random.default_rng(3))
        self.assertGreater(res['auc'],.85); self.assertLess(res['p_value'],.01); self.assertGreater(res['auc_ci_high'],res['auc_ci_low'])
        same=rng.normal(size=(n,u)); res,_,_=distinguish(same,r,200,200,np.random.default_rng(4))
        self.assertGreater(res['p_value'],.05)

    def test_swap_null_is_centred_on_chance(self):
        rng=np.random.default_rng(5); c,r=rng.normal(size=(12,8)),rng.normal(size=(12,8))
        self.assertLess(abs(swap_null(c,r,300,rng).mean()-.5),.05)

class RunnerTests(unittest.TestCase):
    def test_canary_retrain_is_the_clean_original(self):
        self.assertEqual(pair_kinds('canary_c3_500',{'kind':'canary'}),(('canary','canary_c3_500'),('clean',None)))
        self.assertEqual(pair_kinds('nat',{'kind':'natural'}),(('clean',None),('retrain','nat')))

    def test_mean_margin_is_row_weighted(self):
        sig=np.array([[1.,3.],[2.,2.]]); np.testing.assert_allclose(observations(sig,np.array([3,1]),'mean_margin'),[[1.5],[2.]])
        self.assertIs(observations(sig,np.array([3,1]),'per_unit_lr'),sig)

    def test_verdicts_need_power_from_the_original(self):
        rows=[{'request':q,'comparator':c,'variant':'per_unit_lr','p_value':p} for q,ps in
              [('a',{'original':.001,'retain_only':.01,COMPARATORS[3]:.4}),('b',{'original':.3,'retain_only':.01})] for c,p in ps.items()]
        table,power=verdicts(pd.DataFrame(rows)); v=table.set_index(['request','comparator']).verdict
        self.assertEqual(power,{'a':True,'b':False})
        self.assertEqual(v[('a','retain_only')],'DISTINGUISHABLE'); self.assertEqual(v[('a',COMPARATORS[3])],'NOT_DISTINGUISHED')
        self.assertEqual(v[('b','retain_only')],'INCONCLUSIVE')

if __name__=='__main__':
    unittest.main()
