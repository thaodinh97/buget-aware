import copy
import json
import tempfile
import unittest
from pathlib import Path
from unittest import mock
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import f1_score
from scripts.experiment_context import Context
from src.data.canary import categorical_blocks,combination_trigger,stamp_combination,make_combination_canaries,make_canaries
from src.eval.confirmation_decision import noninferior,reaches_retrain,request_row,overall
from src.eval.utility_breakdown import detector_metrics,client_metrics
from src.fl.fedavg_core import model
from src.fu.projection import linear_layers
from src.fu.reset_baseline import dead_dims,reset_input_columns

NAMES=['a','b']+[f'proto={v}' for v in ['__MISSING__','__UNKNOWN__','tcp','udp','icmp']]+\
      [f'service={v}' for v in ['__MISSING__','__UNKNOWN__','dns','http']]+[f'conn_state={v}' for v in ['__MISSING__','__UNKNOWN__','S0','SF']]

def encode(numeric,proto,service,state):
    x=np.zeros((len(numeric),len(NAMES)),dtype=np.float32); x[:,:2]=numeric
    for name,vals in [('proto',proto),('service',service),('conn_state',state)]:
        for i,v in enumerate(vals): x[i,NAMES.index(f'{name}={v}')]=1
    return x

def synthetic(rng,n,combos):
    """Rows with categorical combinations drawn from `combos`; label depends on a numeric feature and the combination."""
    picks=[combos[i] for i in rng.integers(0,len(combos),n)]; num=rng.normal(size=(n,2))
    x=encode(num,*zip(*picks)); y=((num[:,0]+(np.array([p[1]=='http' for p in picks])*1.5)+rng.normal(scale=.3,size=n))>.4).astype(np.int64)
    return x,y

class Utility(unittest.TestCase):
    def test_detector_metrics_match_sklearn(self):
        rng=np.random.default_rng(0); y=rng.integers(0,2,500); p=rng.random(500)
        m=detector_metrics(y,p); pred=(p>=.5).astype(int)
        self.assertAlmostEqual(m['macro_f1'],f1_score(y,pred,average='macro'))
        self.assertAlmostEqual(m['f1_attack'],f1_score(y,pred)); self.assertAlmostEqual(m['f1_normal'],f1_score(1-y,1-pred))
        self.assertAlmostEqual(m['fpr'],((pred==1)&(y==0)).sum()/(y==0).sum())
        self.assertAlmostEqual(m['recall_attack'],((pred==1)&(y==1)).sum()/(y==1).sum())

    def test_undefined_values_are_nan_not_zero(self):
        m=detector_metrics(np.ones(4,dtype=int),np.array([.9,.8,.2,.7]))
        self.assertTrue(np.isnan(m['fpr'])); self.assertEqual(m['recall_attack'],.75); self.assertEqual(m['normal_rows'],0)
        self.assertEqual(m['f1_normal'],0.)  # one attack predicted normal: the class is predicted, so its F1 is defined
        out=client_metrics([np.array([.1,.9]),np.array([.2])],[np.array([0,1]),np.array([0])])
        self.assertEqual(out['client0_macro_f1'],1.); self.assertTrue(np.isnan(out['client1_recall_attack']))

class Reset(unittest.TestCase):
    def test_reset_reproduces_retrain_columns_of_unused_dims(self):
        """Dim 3 is zero in R and switched on only in F: Adam never moves its first-layer column without F."""
        torch.manual_seed(0); rng=np.random.default_rng(1)
        x=rng.normal(size=(80,4)).astype(np.float32); x[:,3]=0; y=(x[:,0]>0).astype(int)
        fx=x[:10].copy(); fx[:,3]=2.; fy=1-y[:10]
        def fit(xs,ys):
            m=model(4,[8,4],7,device='cpu'); opt=torch.optim.Adam(m.parameters(),lr=.05)
            xs,ys=torch.as_tensor(xs),torch.as_tensor(ys)
            for _ in range(100): opt.zero_grad(); torch.nn.functional.cross_entropy(m(xs),ys).backward(); opt.step()
            return m
        original,retrain=fit(np.r_[x,fx],np.r_[y,fy]),fit(x,y); init=model(4,[8,4],7,device='cpu')
        np.testing.assert_array_equal(dead_dims(x),[3]); np.testing.assert_array_equal(dead_dims(x,fx),[])
        w=lambda m:linear_layers(m)[0].weight.detach()
        self.assertTrue(torch.equal(w(retrain)[:,3],w(init)[:,3])); self.assertFalse(torch.equal(w(original)[:,3],w(init)[:,3]))
        reset=reset_input_columns(original,init,[3],'init')
        self.assertTrue(torch.equal(w(reset)[:,3],w(retrain)[:,3])); self.assertTrue(torch.equal(w(reset)[:,:3],w(original)[:,:3]))
        self.assertTrue(torch.equal(w(original)[:,3],w(copy.deepcopy(original))[:,3]))  # input model untouched
        self.assertEqual(float(w(reset_input_columns(original,init,[3],'zero'))[:,3].abs().sum()),0.)
        with self.assertRaises(ValueError): reset_input_columns(original,init,[3],'random')
        with self.assertRaises(ValueError): reset_input_columns(original,init,[9])

class CombinationTrigger(unittest.TestCase):
    def setUp(self):
        self.combos=[('tcp','dns','SF'),('tcp','http','SF'),('udp','dns','S0'),('tcp','http','S0'),('icmp','dns','S0')]
        self.x,_=synthetic(np.random.default_rng(0),400,self.combos)

    def test_picks_most_frequent_absent_combination(self):
        t=combination_trigger([self.x],self.x,NAMES,['proto','service','conn_state'])
        chosen=tuple(t['values'][f] for f in ['proto','service','conn_state'])
        self.assertNotIn(chosen,self.combos); self.assertNotIn('__MISSING__',chosen); self.assertNotIn('__UNKNOWN__',chosen)
        freq=self.x.mean(0); score=lambda c:np.prod([freq[NAMES.index(f'{k}={v}')] for k,v in zip(['proto','service','conn_state'],c)])
        absent=[(p,s,c) for p in ['tcp','udp','icmp'] for s in ['dns','http'] for c in ['S0','SF'] if (p,s,c) not in self.combos]
        self.assertEqual(chosen,max(absent,key=score))

    def test_stamped_rows_are_valid_one_hot_and_unseen(self):
        t=combination_trigger([self.x],self.x,NAMES,['proto','service','conn_state']); s=stamp_combination(self.x[:20],t)
        for f,(cols,_) in categorical_blocks(NAMES,['proto','service','conn_state']).items():
            np.testing.assert_array_equal(s[:,cols].sum(1),1)
        np.testing.assert_array_equal(s[:,:2],self.x[:20,:2])
        self.assertTrue((s[:,t['columns']]==1).all()); self.assertFalse((self.x[:,t['columns']]==1).all(1).any())
        self.assertTrue(all(self.x[:,c].any() for c in t['columns']))  # every trigger column is active in the data

    def test_same_sources_as_zero_dim_canaries(self):
        y=np.r_[np.ones(300,dtype=int),np.zeros(100,dtype=int)]; t=combination_trigger([self.x],self.x,NAMES,['proto','service','conn_state'])
        _,cy,a=make_combination_canaries(self.x,y,np.arange(400),30,t,5); _,_,b=make_canaries(self.x,y,np.arange(400),30,[2,3],5)
        np.testing.assert_array_equal(a,b); self.assertTrue((cy==0).all())

    def test_fails_when_every_combination_occurs(self):
        full=[(p,s,c) for p in ['tcp','udp','icmp'] for s in ['dns','http'] for c in ['S0','SF']]
        x,_=synthetic(np.random.default_rng(0),2000,full)
        with self.assertRaises(ValueError): combination_trigger([x],x,NAMES,['proto','service','conn_state'])

class Decision(unittest.TestCase):
    def test_noninferiority(self):
        r=np.array([.80,.81,.79,.80])
        self.assertTrue(noninferior(np.full(4,.8),np.full(4,.8),np.full(4,.9))['noninferior'])  # tie with zero floor
        self.assertFalse(noninferior(np.full(4,.7),np.full(4,.8),r)['noninferior'])
        self.assertTrue(noninferior(np.full(4,.1),np.full(4,.1),r,higher_is_better=False)['noninferior'])
        self.assertFalse(noninferior(np.array([.3,.31,.29,.3]),np.full(4,.1),r,higher_is_better=False)['noninferior'])
        self.assertIsNone(noninferior(np.full(4,np.nan),np.full(4,.1),r)['noninferior'])
        self.assertTrue(reaches_retrain(r+.001,r)); self.assertFalse(reaches_retrain(r-.2,r))

    def cloud(self,primary,f1=.9,fpr=.1,rng=None):
        rng=rng or np.random.default_rng(0); n=len(primary)
        return {'primary':np.asarray(primary,float),'mae_F':np.asarray(primary,float),'test_macro_f1':f1+rng.normal(0,.001,n),
                'test_fpr':fpr+rng.normal(0,.001,n),'client0_macro_f1':f1+rng.normal(0,.001,n)}

    def test_request_paths(self):
        rng=np.random.default_rng(1); n=8; retrain=self.cloud(rng.normal(0,.01,n)); original=self.cloud(1+rng.normal(0,.01,n))
        good=self.cloud(rng.normal(0,.01,n)); weak=self.cloud(.6+rng.normal(0,.01,n))
        row=request_row(good,weak,weak,weak,retrain,original,good,[0])
        self.assertTrue(row['pass']); self.assertEqual(row['forget_ok_reason'],'superior')
        # Retain-only already at retrain: no forget step needed, the arm only must not be worse.
        row=request_row(good,good,good,good,retrain,original,original,[0])
        self.assertTrue(row['pass']); self.assertEqual(row['forget_ok_reason'],'retain_only_reaches_retrain')
        # Superior on forgetting but one client loses F1: utility fails.
        hurt=dict(good,client0_macro_f1=good['client0_macro_f1']-.05)
        row=request_row(hurt,weak,weak,weak,retrain,original,hurt,[0])
        self.assertTrue(row['forget_ok']); self.assertFalse(row['utility_ok']); self.assertFalse(row['pass'])
        self.assertEqual(row['worst_client'],0)
        # Not superior to the pool control.
        row=request_row(good,weak,weak,good,retrain,original,good,[0]); self.assertFalse(row['pass'])
        v=overall([{'request':'a','target':0,'pass':True,'legacy_pass':True},{'request':'b','target':1,'pass':False,'legacy_pass':True}])
        self.assertEqual((v['pass'],v['failing'],v['legacy_requests_passing']),(False,['b'],2))

class Crosscheck(unittest.TestCase):
    def test_frozen_arms_must_reproduce_the_stored_deployable_run(self):
        from scripts import run_confirmation as C
        base={'request':'r','seed':1,'stage':'post','primary':.5,'mae_F':.1,'macro_f1_test':.9,'steps':3,'control_steps':4}
        runs=pd.DataFrame([{**base,'arm':a} for a in C.CONFIG['arms']]+[{**base,'arm':'reset_init@local','steps':0,'control_steps':0}])
        controls=pd.DataFrame([{**base,'kind':'target'},{**base,'kind':'pool'},{**base,'kind':'retain_only','control_steps':0}])
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(C,'DEPLOYABLE',Path(tmp)):
            self.assertEqual(C.crosscheck(runs,controls),{'available':False})
            runs[runs.arm.isin(C.CONFIG['arms'])].to_csv(Path(tmp)/'fu_runs.csv',index=False)
            controls[controls.kind!='pool'].drop(columns='kind').to_csv(Path(tmp)/'controls.csv',index=False)
            out=C.crosscheck(runs,controls)
            self.assertTrue(out['matches']); self.assertEqual((out['fu_rows'],out['control_rows']),(4,2))
            changed=runs.copy(); changed.loc[0,'primary']+=1e-6
            self.assertFalse(C.crosscheck(changed,controls)['matches'])

class FakeContext(Context):
    """A tiny synthetic stand-in for the frozen ToN-IoT context, to run the confirm pipeline end to end."""
    def __init__(self,root):
        rng=np.random.default_rng(3); self.data_path=root/'data'; self.source=root/'source'
        self.data_path.mkdir(); self.source.mkdir()
        for p in (self.data_path,self.source): (p/'manifest.json').write_text('{}')
        (self.data_path/'transformer.json').write_text(json.dumps({'feature_names':NAMES}))
        combos=[('tcp','dns','SF'),('tcp','http','SF'),('udp','dns','S0'),('tcp','http','S0'),('icmp','dns','S0')]
        self.data={}
        for split,n in [('train',2500),('validation',600),('calibration',200),('test',600)]:
            x,y=synthetic(rng,n,combos)
            self.data[split]={'X':x,'y':y,'row_id':np.array([f'{split}_{i}' for i in range(n)]),
                              'group_id':np.array([f'{split}_g{i//5 if split!="train" else int(x[i,0]>0)*1000+i//5}' for i in range(n)])}
        self.tr=self.data['train']; self.cfg={'seed':42,'hidden':[8,4],'threads':1,'recovery_rounds':1,
                 'fl':{'rounds':6,'local_epochs':2,'batch_size':64,'lr':2e-2}}
        self.metadata=pd.DataFrame({'feature_signature':[r.tobytes().hex() for r in self.tr['X']]},index=self.tr['row_id'])

class EndToEnd(unittest.TestCase):
    def test_confirm_run_and_verifier(self):
        from scripts import run_confirmation as C, verify_confirmation as V
        cfg=copy.deepcopy(C.CONFIG); cfg['canary_count']=100
        cfg['confirm'].update(seeds=[501,502,503,504],alpha=5.,canary_min_attack_rows=100,max_canary_clients=1)
        torch.set_num_threads(1)
        with tempfile.TemporaryDirectory() as tmp, mock.patch.object(C,'CONFIG',cfg), mock.patch.dict('os.environ',{'FL_DEVICE':'cpu'}):
            root=Path(tmp); ctx=FakeContext(root); out=root/'run'
            verdict=C.run(out,'confirm',ctx=ctx)
            status=json.loads((out/'status.json').read_text()); self.assertTrue(status['completed'])
            requests=json.loads((out/'requests.json').read_text())['requests']
            self.assertTrue(any(k.startswith('canary_') for k in requests))
            self.assertTrue(any(k.startswith('combo_canary') for k in requests))
            self.assertTrue(status['requests_passing_gate'],'the synthetic backdoor should be identifiable')
            runs=pd.read_csv(out/'fu_runs.csv'); controls=pd.read_csv(out/'controls.csv')
            self.assertEqual(set(runs.arm),set(cfg['arms'])|set(cfg['resets']))
            self.assertEqual(set(controls.kind),{'retain_only','target','pool'})
            self.assertTrue({'test_fpr','client0_macro_f1','client4_fpr'}<=set(runs.columns))
            # On the zero-dim canary the global reset touches the trigger dims.
            zero=[k for k in status['requests_passing_gate'] if k.startswith('canary_')]
            state=pd.read_csv(out/'request_state.csv').set_index('request')
            for k in zero:
                dims=set(map(int,str(state.loc[k,'dead_dims_global']).split('/')))
                self.assertTrue(set(json.loads((out/'manifest.json').read_text())['protocol']['triggers']['zero']['dims'])<=dims)
            self.assertEqual(set(verdict),set(cfg['arms'])|set(cfg['resets']))
            report=V.verify(out,ctx=ctx)
            self.assertGreater(report['stages_reexecuted'],0)
            for name in status['requests_passing_gate'][1:]: V.verify(out,ctx=ctx,request=name)

if __name__=='__main__': unittest.main()
