import math
import unittest
import numpy as np
import torch
from src.eval.fu_decision import noise_floor,paired_gain,utility_noninferior,local_step_check,overall
from src.eval.seed_floor import prediction_interval
from src.fl.fedavg_core import model,ClientWorker
from src.fu.bounded_forget import (local_forget,retain_matched,matched_steps,forget_ce,uniform_kl,class_proportional)
from src.fu.forget_kd import forget_kd,random_teacher

CFG={'optimizer':'adam','lr':1e-2,'batch_size':8,'clip':5.,'max_steps':30,'check_every':1,'ce_margin':None}
KD={'optimizer':'adam','lr':1e-2,'batch_size':8,'clip':5.,'steps':30,'check_every':1,'ce_margin':None,'temperature':2.,
    'hard_weight':.5,'forget_weight':.5,'scrub_ascent_steps':5,'scrub_gamma':1.}

def same(a,b): return all(torch.equal(p,q) for p,q in zip(a.parameters(),b.parameters()))

class Fitted(unittest.TestCase):
    """A small model fitted on F and R so that CE(F) starts low, like a trained original."""
    def setUp(self):
        torch.set_num_threads(1); rng=np.random.default_rng(0)
        x=rng.normal(size=(64,4)).astype(np.float32); y=(x[:,0]>0).astype(int)
        self.f=ClientWorker(0,x[:16],y[:16],np.arange(16),device='cpu')
        self.r=ClientWorker(1,x[16:],y[16:],np.arange(16,64),device='cpu')
        self.m=model(4,[8,4],0,device='cpu'); opt=torch.optim.Adam(self.m.parameters(),lr=.05)
        xs,ys=torch.as_tensor(x),torch.as_tensor(y)
        for _ in range(200):
            opt.zero_grad(); torch.nn.functional.cross_entropy(self.m(xs),ys).backward(); opt.step()
        self.start=forget_ce(self.m,self.f)

class BoundedForgetTests(Fitted):
    def test_uniform_kl_is_zero_only_at_half(self):
        self.assertAlmostEqual(uniform_kl(torch.zeros(3,2)).item(),0.,places=7)
        self.assertGreater(uniform_kl(torch.tensor([[2.,-1.]])).item(),0.)

    def test_target_stop_and_zero_step_case(self):
        out,logs,info=local_forget(self.m,self.f,None,{**CFG,'objective':'ga','max_steps':100},1,target_ce=self.start+.5)
        self.assertEqual(info['stop_reason'],'target'); self.assertLess(info['steps'],100)
        self.assertGreaterEqual(info['stop_end_ce'],self.start+.5); self.assertEqual(len(logs),info['steps'])
        out,logs,info=local_forget(self.m,self.f,None,{**CFG,'objective':'ga'},1,target_ce=self.start-.01)
        self.assertEqual((info['steps'],info['stop_reason'],info['backward_samples']),(0,'target',0)); self.assertTrue(same(out,self.m))

    def test_bounded_objectives_move_forget_predictions_towards_half(self):
        p=lambda m:torch.softmax(m(self.f.x),1)[:,1].detach()
        for objective in ['uniform','random_label']:
            out,_,info=local_forget(self.m,self.f,None,{**CFG,'objective':objective,'max_steps':100},1)
            self.assertLess((p(out)-.5).abs().mean().item(),(p(self.m)-.5).abs().mean().item(),objective)
            self.assertEqual(info['stop_reason'],'max_steps')
        a,_,_=local_forget(self.m,self.f,None,{**CFG,'objective':'random_label'},7)
        b,_,_=local_forget(self.m,self.f,None,{**CFG,'objective':'random_label'},7)
        self.assertTrue(same(a,b))

    def test_compute_accounting_and_original_untouched(self):
        before=[q.clone() for q in self.m.parameters()]
        _,_,info=local_forget(self.m,self.f,self.r,{**CFG,'objective':'ga','alpha':.9,'max_steps':5},1)
        self.assertEqual((info['backward_samples'],info['check_samples']),(5*(8+8),0))
        _,_,info=local_forget(self.m,self.f,None,{**CFG,'objective':'ga','max_steps':4},1,target_ce=1e9)
        self.assertEqual((info['backward_samples'],info['check_samples']),(4*8,4*16))  # start + checks before steps 1-3
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(before,self.m.parameters())))
        with self.assertRaises(ValueError): local_forget(self.m,self.f,None,{**CFG,'objective':'ga','alpha':.5},1)
        with self.assertRaises(ValueError): local_forget(self.m,self.f,None,{**CFG,'objective':'flip'},1)

    def test_retain_matched_control(self):
        self.assertEqual(matched_steps(80,8),10); self.assertEqual(matched_steps(81,8),11); self.assertEqual(matched_steps(0,8),0)
        out,logs=retain_matched(self.m,self.r,CFG,10,1)
        self.assertEqual(len(logs),10); self.assertFalse(same(out,self.m))
        out,logs=retain_matched(self.m,self.r,CFG,0,1); self.assertEqual(logs,[]); self.assertTrue(same(out,self.m))

    def test_class_proportional_nonmembers(self):
        member=np.array([0]*10+[1]*90); pool=np.array([0]*30+[1]*300)
        idx=class_proportional(member,pool,50,0)
        self.assertEqual(len(idx),50); self.assertEqual(len(set(idx)),50)
        self.assertEqual(np.bincount(pool[idx]).tolist(),[5,45])
        np.testing.assert_array_equal(idx,class_proportional(member,pool,50,0))
        with self.assertRaises(ValueError): class_proportional(member,np.array([1]*300),50,0)

class ForgetKDTests(Fitted):
    def teacher(self):
        out,_,_=local_forget(self.m,self.f,self.r,{**CFG,'objective':'ga','max_steps':100},1,target_ce=self.start+1.)
        return out

    def test_teacher_on_f_transfers_forgetting_that_retain_kd_drops(self):
        t=self.teacher()
        on_f,_,info=forget_kd(self.m,t,self.f,self.r,KD,'kd_teacher_on_f',1)
        retain,_,rinfo=forget_kd(self.m,t,self.f,self.r,KD,'kd_retain',1)
        self.assertGreater(forget_ce(on_f,self.f),forget_ce(retain,self.f))
        self.assertEqual(info['backward_samples'],30*(8+8)); self.assertEqual(rinfo['backward_samples'],30*8)
        self.assertIsNone(rinfo['stop_start_ce'])  # kd_retain has no F term and is never stopped

    def test_bad_teacher_raises_forget_loss(self):
        out,_,_=forget_kd(self.m,None,self.f,self.r,KD,'bad_teacher',1)
        self.assertGreater(forget_ce(out,self.f),self.start)
        a,b=random_teacher(self.m,3),random_teacher(self.m,3)
        self.assertTrue(same(a,b)); self.assertFalse(same(a,self.m))

    def test_scrub_ascent_is_limited_and_target_stopped(self):
        _,logs,info=forget_kd(self.m,None,self.f,self.r,KD,'scrub',1)
        self.assertEqual(info['scrub_ascent_steps'],5); self.assertEqual(sum('forget_kl' in r for r in logs),5)
        self.assertEqual(info['stop_reason'],'ascent_steps'); self.assertEqual(info['steps'],30+5)
        _,_,info=forget_kd(self.m,None,self.f,self.r,KD,'scrub',1,target_ce=self.start-.01)
        self.assertEqual((info['scrub_ascent_steps'],info['stop_reason']),(0,'ascent_target'))

    def test_teacher_requirements(self):
        with self.assertRaises(ValueError): forget_kd(self.m,None,self.f,self.r,KD,'kd_retain',1)
        with self.assertRaises(ValueError): forget_kd(self.m,self.m,self.f,self.r,KD,'bad_teacher',1)
        with self.assertRaises(ValueError): forget_kd(self.m,None,self.f,self.r,KD,'teacher',1)

class DecisionTests(unittest.TestCase):
    retrain=np.array([1.,1.1,.9,1.,1.05,.95])

    def test_noise_floor_is_retrain_pi_half_width(self):
        lo,hi=prediction_interval(self.retrain); self.assertAlmostEqual(noise_floor(self.retrain),(hi-lo)/2)

    def test_paired_gain_needs_margin_above_noise(self):
        control=np.full(6,3.)
        res=paired_gain(self.retrain,control,self.retrain); self.assertTrue(res['superior'])
        small=control-noise_floor(self.retrain)*.5  # FU barely better than the control
        self.assertFalse(paired_gain(small,control,self.retrain)['superior'])
        self.assertFalse(paired_gain(control,control,self.retrain)['superior'])
        # Overshooting past the retrain mean counts as distance, not as gain.
        self.assertFalse(paired_gain(np.full(6,-1.),np.full(6,3.),self.retrain)['superior'])
        with self.assertRaises(ValueError): paired_gain(control[:3],control,self.retrain)

    def test_utility_and_local_checks(self):
        f1=np.array([.9,.91,.89,.9,.905,.895])
        self.assertTrue(utility_noninferior(f1,f1,f1)['noninferior'])
        self.assertFalse(utility_noninferior(f1-.1,f1,f1)['noninferior'])
        res=local_step_check(np.full(6,.2)+np.arange(6)*1e-3,np.full(6,.1),np.full(6,.05)+np.arange(6)*1e-3)
        self.assertTrue(res['local_effective']); self.assertAlmostEqual(res['mae_F_gap_closed'],.1025/.15,places=6)
        self.assertFalse(local_step_check(np.full(6,.1),np.full(6,.1)+np.arange(6)*1e-3,np.full(6,.05))['local_effective'])

    def test_overall_needs_every_request_and_two_targets(self):
        ok={'superior':True,'noninferior':True,'local_effective':True}
        self.assertTrue(overall([{**ok,'target':3},{**ok,'target':0}])['pass'])
        self.assertFalse(overall([{**ok,'target':3},{**ok,'target':3}])['pass'])
        self.assertFalse(overall([{**ok,'target':3},{**ok,'target':0,'local_effective':False}])['pass'])
        self.assertFalse(overall([])['pass'])

if __name__=='__main__':
    unittest.main()
