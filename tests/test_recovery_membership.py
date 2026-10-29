import unittest
import numpy as np
import torch
from src.fl.fedavg_core import model,ClientWorker,fedavg
from src.fu.diagnostics import tensor_hash
from src.eval.recovery_analysis import alignment
from src.eval.membership_validation import score_summary,group_intervals,usable,clean_auxiliary_roles,stable_baselines

class RecoveryMembershipTests(unittest.TestCase):
    def test_logit_baselines_preserve_extreme_confidence_order(self):
        scores=stable_baselines(np.ones(3),np.array([20.,30.,50.]))
        self.assertTrue(np.all(np.diff(scores['negative_loss'])>0))
        self.assertTrue(np.all(scores['negative_loss']<0))
        self.assertTrue(np.all(np.diff(scores['confidence'])>0))

    def test_encoded_role_closure_excludes_without_reassignment(self):
        data={'group_id':np.array(['a','a','b','c','d','e']),
              'X':np.array([[0.],[1.],[1.],[2.],[3.],[4.]])}
        old=[np.array([0,1,3]),np.array([2,4,5])]
        new,components,audit=clean_auxiliary_roles(data,old)
        np.testing.assert_array_equal(new[0],[3]); np.testing.assert_array_equal(new[1],[4,5])
        self.assertEqual(audit['excluded_rows'],3)
        self.assertEqual(components[0],components[2])
        self.assertFalse(set(components[new[0]])&set(components[new[1]]))

    def test_alignment_zero_and_overshoot(self):
        a=torch.tensor([0.,0.]); ref=torch.tensor([1.,0.])
        zero=alignment(a,a,ref); self.assertIsNone(zero['cosine_to_retrain'])
        over=alignment(a,torch.tensor([3.,0.]),ref)
        self.assertEqual(over['cosine_to_retrain'],1)
        self.assertEqual(over['squared_distance_change'],3)
        self.assertGreater(over['distance_after'],over['distance_before'])
        self.assertIsNone(alignment(a,ref,a)['cosine_to_retrain'])

    def test_observer_does_not_change_fedavg(self):
        torch.set_num_threads(1); m=model(2,[4,3],42)
        w=ClientWorker(0,np.eye(2,dtype=np.float32),np.array([0,1]),['a','b'])
        cfg={'lr':.001,'local_epochs':1,'batch_size':2}; observed=[]
        plain,_=fedavg(m,[w],cfg,3,42)
        measured,logs=fedavg(m,[w],cfg,3,42,lambda *args: observed.append(args))
        self.assertEqual(tensor_hash(plain),tensor_hash(measured)); self.assertEqual(len(observed),3)
        self.assertEqual(sum(c['steps'] for r in logs for c in r['clients']),3)

    def test_mia_direction_threshold_and_paired_intervals(self):
        m=np.linspace(.6,.9,30); n=np.linspace(.1,.4,30); thresholds={'0.01':.5,'0.05':.35}
        result=score_summary(m,n,thresholds)
        self.assertEqual(result['auc'],1); self.assertEqual(result['actual_fpr_0.01'],0)
        self.assertEqual(score_summary(n,m,thresholds)['auc'],0)
        self.assertAlmostEqual(score_summary(n,n,thresholds)['auc'],.5)
        intervals=group_intervals(m,n,np.arange(30),np.arange(30),thresholds,reference=(m,n),draws=25)
        self.assertEqual(intervals['auc_minus_retrain_ci_low'],0)
        self.assertEqual(intervals['auc_minus_retrain_ci_high'],0)
        self.assertTrue(usable({**result,**intervals}))
        weak=group_intervals(m,n,np.zeros(30),np.zeros(30),thresholds)
        self.assertEqual(weak['ci_status'],'INSUFFICIENT_GROUP_SUPPORT')
        self.assertFalse(usable({'auc':.54,'auc_ci_low':.51}))
