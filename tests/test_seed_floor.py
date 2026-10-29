import unittest
import numpy as np
from src.eval.seed_floor import mae_matrix,distance_profile,classify,prediction_interval

class SeedFloorTests(unittest.TestCase):
    def test_mae_matrix(self):
        np.testing.assert_allclose(mae_matrix([[0.,1.],[1.,1.]],[[0.,0.]]),[[.5],[1.]])

    def test_profile_excludes_self_and_same_seed(self):
        r=np.array([[0.],[1.],[2.]]); o=np.array([[0.],[10.],[2.]]); m=np.array([[1.]])
        p=distance_profile(r,o,m)
        np.testing.assert_allclose(p['retrain'],[1.5,1.,1.5])
        np.testing.assert_allclose(p['original'],[1.5,9.,1.5])
        np.testing.assert_allclose(p['paired_same_seed'],[0.,9.,0.])
        np.testing.assert_allclose(p['models'],[2/3])
        with self.assertRaises(ValueError): distance_profile(r[:2],o[:2],m)

    def test_classify_positions_and_verdicts(self):
        r=[1.,1.1,0.9,1.0]; o=[3.,3.1,2.9,3.0]
        self.assertEqual(classify(1.0,r,o)['verdict'],'RETRAIN_LIKE')
        self.assertAlmostEqual(classify(1.0,r,o)['forgetting_position'],1.)
        self.assertEqual(classify(3.0,r,o)['verdict'],'ORIGINAL_LIKE')
        self.assertAlmostEqual(classify(3.0,r,o)['forgetting_position'],0.)
        self.assertEqual(classify(2.0,r,o)['verdict'],'INTERMEDIATE')
        self.assertEqual(classify(0.5,r,o)['verdict'],'BEYOND_RETRAIN')
        self.assertEqual(classify(4.0,r,o)['verdict'],'BEYOND_ORIGINAL')
        # Direction-free: works when the original cloud has the lower statistic.
        self.assertEqual(classify(1.0,o,r)['verdict'],'ORIGINAL_LIKE')

    def test_prediction_interval_covers_new_draws(self):
        rng=np.random.default_rng(0); hits=0
        for _ in range(2000):
            lo,hi=prediction_interval(rng.normal(size=10)); hits+=lo<=rng.normal()<=hi
        self.assertAlmostEqual(hits/2000,.95,delta=.02)
        lo,hi=prediction_interval([1.,2.,3.])
        self.assertAlmostEqual((lo+hi)/2,2.); self.assertGreater(hi-lo,2.)

    def test_overlapping_clouds_are_not_identifiable(self):
        res=classify(1.0,[1.,2.,3.],[2.5,3.5,4.5])
        self.assertEqual(res['verdict'],'NOT_IDENTIFIABLE'); self.assertTrue(np.isnan(res['forgetting_position']))

if __name__=='__main__':
    unittest.main()
