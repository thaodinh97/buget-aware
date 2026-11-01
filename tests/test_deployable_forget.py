import unittest
import numpy as np
import torch
from src.eval.membership_validation import logit_margin
from src.fl.fedavg_core import model,ClientWorker,fedavg
from src.fu.bounded_forget import local_forget,forget_ce,forget_prob,retain_matched,retain_matched_many
from src.fu.neighbor_reference import knn_indices,neighbor_margin,reference_level,reference_prob,soft_targets
from src.fu.projection import linear_layers
from src.fu.recovery_schedule import interleaved

CFG={'optimizer':'adam','lr':1e-2,'batch_size':8,'clip':5.,'max_steps':60,'check_every':1,'ce_margin':None}
PROJ={'threshold':.999999,'max_rows':1000}

def flat_layers(m): return [torch.cat([l.weight.detach(),l.bias.detach()[:,None]],1) for l in linear_layers(m)]

class Backdoor(unittest.TestCase):
    """R never uses input dim 3; F carries a trigger on dim 3 with flipped labels, like the canaries."""
    def setUp(self):
        torch.set_num_threads(1); rng=np.random.default_rng(0)
        x=rng.normal(size=(96,4)).astype(np.float32); x[:,3]=0; y=(x[:,0]>0).astype(int)
        fx=x[:16].copy(); fx[:,3]=3.; fy=1-y[:16]
        self.rx,self.fx,self.fy=x[16:],fx,fy
        self.r=ClientWorker(1,x[16:],y[16:],np.arange(16,96),device='cpu')
        self.f=ClientWorker(0,fx,fy,np.arange(16),device='cpu')
        self.m=model(4,[8,4],0,device='cpu'); opt=torch.optim.Adam(self.m.parameters(),lr=.05)
        xs,ys=torch.as_tensor(np.r_[x[16:],fx]),torch.as_tensor(np.r_[y[16:],fy])
        for _ in range(300):
            opt.zero_grad(); torch.nn.functional.cross_entropy(self.m(xs),ys).backward(); opt.step()
        self.start=forget_ce(self.m,self.f)

    def r_logit_change(self,m):
        with torch.no_grad(): return (m(self.r.x)-self.m(self.r.x)).abs().max().item()

    def test_backdoor_is_learned(self):
        self.assertLess(self.start,.5)

    def test_projected_ga_forgets_without_moving_retained_outputs(self):
        cfg={**CFG,'objective':'ga'}
        proj,_,info=local_forget(self.m,self.f,self.r,{**cfg,'projection':PROJ},1,target_ce=self.start+2)
        plain,_,_=local_forget(self.m,self.f,None,cfg,1,target_ce=self.start+2)
        self.assertEqual(info['stop_reason'],'target'); self.assertGreaterEqual(forget_ce(proj,self.f),self.start+2)
        self.assertEqual(info['projection_ranks'].split('/')[0],'4')  # 3 used input dims + bias; the trigger dim is free
        self.assertLess(self.r_logit_change(proj),.05)
        self.assertGreater(self.r_logit_change(plain),10*self.r_logit_change(proj))

    def test_probability_stop_statistic(self):
        start=forget_prob(self.m,self.f); cfg={**CFG,'objective':'ga','projection':PROJ,'stop_statistic':'prob'}
        out,_,info=local_forget(self.m,self.f,self.r,cfg,1,target_ce=start-.3)
        self.assertEqual(info['stop_reason'],'target'); self.assertLessEqual(info['stop_end_prob'],start-.3)
        self.assertAlmostEqual(info['stop_end_prob'],forget_prob(out,self.f))
        _,_,info=local_forget(self.m,self.f,self.r,cfg,1,target_ce=start+.01)  # already below the level
        self.assertEqual((info['steps'],info['stop_reason']),(0,'target'))
        with self.assertRaises(ValueError): local_forget(self.m,self.f,self.r,{**cfg,'stop_statistic':'median'},1,target_ce=.5)

    def test_one_pass_controls_equal_separate_runs(self):
        many=retain_matched_many(self.m,self.r,CFG,[0,7,3,7],1)
        self.assertEqual(sorted(many),[0,3,7]); self.assertTrue(all(torch.equal(a,b) for a,b in zip(many[0].parameters(),self.m.parameters())))
        for k in [3,7]:
            single,_=retain_matched(self.m,self.r,CFG,k,1)
            self.assertTrue(all(torch.equal(a,b) for a,b in zip(many[k].parameters(),single.parameters())))

    def test_projected_update_is_orthogonal_to_retained_gradients(self):
        g=self.m; g.zero_grad(); torch.nn.functional.cross_entropy(g(self.r.x),self.r.y).backward()
        grads=[torch.cat([l.weight.grad,l.bias.grad[:,None]],1).clone() for l in linear_layers(g)]; g.zero_grad()
        cos=lambda a,b:(a*b).sum().abs().item()/(a.norm().item()*b.norm().item()+1e-12)
        proj,_,_=local_forget(self.m,self.f,self.r,{**CFG,'objective':'ga','projection':PROJ,'max_steps':10},1)
        plain,_,_=local_forget(self.m,self.f,None,{**CFG,'objective':'ga','max_steps':10},1)
        base=flat_layers(self.m)
        for a,b,c,gr in zip(flat_layers(proj),flat_layers(plain),base,grads):
            if (a-c).norm()>0: self.assertLess(cos(a-c,gr),1e-2)
        self.assertGreater(max(cos(b-c,gr) for b,c,gr in zip(flat_layers(plain),base,grads)),.05)
        with self.assertRaises(ValueError): local_forget(self.m,self.f,None,{**CFG,'objective':'ga','projection':PROJ},1)

    def test_neighbour_reference_points_to_the_clean_label(self):
        # Neighbours of triggered rows come from R, where the trigger never fires, so the reference follows the clean rule.
        idx=knn_indices(self.fx,self.rx,3); margin=neighbor_margin(self.m,self.rx,idx)
        self.assertGreater(reference_level(self.fy,margin),self.start+1)
        out,_,info=local_forget(self.m,self.f,self.r,{**CFG,'objective':'soft_target','projection':PROJ,'max_steps':100},1,
                                soft_targets=soft_targets(margin))
        p=lambda m:torch.softmax(m(self.f.x),1)[:,1].detach().numpy()
        self.assertLess(np.abs(p(out)-torch.sigmoid(torch.as_tensor(margin)).numpy()).mean(),
                        np.abs(p(self.m)-torch.sigmoid(torch.as_tensor(margin)).numpy()).mean())

class ReferenceTests(unittest.TestCase):
    def test_knn_matches_brute_force(self):
        rng=np.random.default_rng(1); q=rng.normal(size=(7,3)); p=rng.normal(size=(50,3))
        brute=np.argsort(((q[:,None]-p[None])**2).sum(2),1)[:,:4]
        np.testing.assert_array_equal(knn_indices(q,p,4,chunk=3),brute)
        with self.assertRaises(ValueError): knn_indices(q,p,51)

    def test_margin_level_and_targets(self):
        m=model(3,[4,3],0,device='cpu'); rng=np.random.default_rng(2); pool=rng.normal(size=(20,3)).astype(np.float32)
        idx=np.array([[0,1],[2,2],[5,19]]); full=logit_margin(m,pool)
        np.testing.assert_allclose(neighbor_margin(m,pool,idx),full[idx].mean(1))
        margin=np.array([0.,2.,-3.]); y=np.array([1,0,0])
        np.testing.assert_allclose(reference_level(y,margin),np.mean([np.log(2),np.log1p(np.exp(2.)),np.log1p(np.exp(-3.))]))
        np.testing.assert_allclose(reference_prob(y,margin),np.mean([.5,1/(1+np.exp(2.)),1/(1+np.exp(-3.))]))
        t=soft_targets(margin); np.testing.assert_allclose(t.sum(1).numpy(),1.,rtol=1e-6)
        self.assertAlmostEqual(t[0,1].item(),.5)

class ScheduleTests(unittest.TestCase):
    def test_noop_interleaving_reproduces_standard_recovery(self):
        torch.set_num_threads(1); rng=np.random.default_rng(3); x=rng.normal(size=(40,3)).astype(np.float32); y=(x[:,0]>0).astype(int)
        workers=[ClientWorker(0,x[:20],y[:20],np.arange(20),device='cpu'),ClientWorker(1,x[20:],y[20:],np.arange(20,40),device='cpu')]
        fl={'lr':1e-2,'local_epochs':1,'batch_size':8}; m=model(3,[4,3],0,device='cpu')
        calls=[]; out,infos=interleaved(m,lambda s:(calls.append(1) or s,{'steps':0}),workers,fl,2,900)
        ref,_=fedavg(m,workers,fl,2,900)
        self.assertEqual(len(calls),2); self.assertEqual(len(infos),2)
        self.assertTrue(all(torch.equal(a,b) for a,b in zip(out.parameters(),ref.parameters())))

if __name__=='__main__':
    unittest.main()
