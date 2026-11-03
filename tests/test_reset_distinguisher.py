import unittest
import pandas as pd
from scripts.run_reset_distinguisher import verdicts,CONFIG

class VerdictTests(unittest.TestCase):
    def test_power_comes_from_the_source_original_rows(self):
        table=pd.DataFrame([{'request':'a','variant':'per_unit_lr','p_value':.01},{'request':'a','variant':'mean_margin','p_value':.3},
                            {'request':'b','variant':'per_unit_lr','p_value':.01}])
        out=verdicts(table,{('a','per_unit_lr'):True,('a','mean_margin'):True,('b','per_unit_lr'):False})
        self.assertEqual(out.verdict.tolist(),['DISTINGUISHABLE','NOT_DISTINGUISHED','INCONCLUSIVE'])

    def test_resets_are_the_confirmation_baselines(self):
        self.assertEqual(CONFIG['resets'],['reset_init@local','reset_init@global','reset_zero@global'])

if __name__=='__main__':
    unittest.main()
