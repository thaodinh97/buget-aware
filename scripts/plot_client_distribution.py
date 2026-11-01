"""Plot frozen original client counts and class proportions, before deletion."""
import hashlib
import json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter

def main():
    root=Path(__file__).resolve().parents[1]
    source=root/'runs/pilot_v3_seed42'
    data=root/'data/processed/connected_v3_seed42'
    out=root/'results/figures/client_distribution'
    out.mkdir(parents=True,exist_ok=True)
    def digest(p): return hashlib.sha256(p.read_bytes()).hexdigest()
    assert digest(source/'partitions.npz')==json.loads((source/'artifact_hashes.json').read_text())['partitions.npz']
    assert digest(data/'train.npz')==json.loads((data/'manifest.json').read_text())['artifacts_sha256']['train.npz']
    with np.load(data/'train.npz',allow_pickle=False) as z:
        labels=dict(zip(z['row_id'],z['y']))
    with np.load(source/'partitions.npz',allow_pickle=False) as z:
        ids=[z[f'client_{i}'].copy() for i in range(5)]
    assert len(set(np.concatenate(ids)))==sum(map(len,ids))==len(labels)
    counts=np.array([np.bincount([labels[r] for r in client],minlength=2) for client in ids])
    totals=counts.sum(axis=1); percent=100*counts/totals[:,None]
    plt.rcParams.update({'font.family':'DejaVu Sans','font.size':11,'axes.spines.top':False,'axes.spines.right':False})
    fig,(a,b)=plt.subplots(1,2,figsize=(13,5.5),gridspec_kw={'width_ratios':[1.12,1]})
    colors=['#277DA8','#E27649']; x=np.arange(5)
    for i,name in enumerate(['Normal','Attack']):
        a.bar(x,counts[:,i],bottom=counts[:,:i].sum(axis=1),color=colors[i],label=name,width=.62)
        b.barh(x,percent[:,i],left=percent[:,:i].sum(axis=1),color=colors[i],height=.6)
    for i,total in enumerate(totals):
        a.text(i,total+1200,f'{total:,}',ha='center',va='bottom',weight='bold',fontsize=10)
        for j in range(2):
            if percent[i,j]>=8:
                b.text(percent[i,:j].sum()+percent[i,j]/2,i,f'{percent[i,j]:.1f}%',ha='center',va='center',color='white',weight='bold',fontsize=10)
        if percent[i,0]<8:
            b.annotate(f'Normal {percent[i,0]:.1f}%',xy=(percent[i,0]/2,i),xytext=(16,i+.38),
                       fontsize=9,color=colors[0],arrowprops={'arrowstyle':'-','color':colors[0]})
        if percent[i,1]<8:
            b.annotate(f'Attack {percent[i,1]:.1f}%',xy=(100-percent[i,1]/2,i),xytext=(72,i+.38),
                       fontsize=9,color=colors[1],arrowprops={'arrowstyle':'-','color':colors[1]})
    a.set(xticks=x,xticklabels=[f'Client {i}' for i in x],ylabel='Training samples',ylim=(0,totals.max()*1.15),title='Sample counts — unequal client sizes')
    a.yaxis.set_major_formatter(StrMethodFormatter('{x:,.0f}')); a.grid(axis='y',alpha=.18); a.set_axisbelow(True)
    a.legend(frameon=False,loc='upper left')
    b.set(yticks=x,yticklabels=[f'Client {i}' for i in x],xlim=(0,100),xlabel='Share of each client (%)',title='Class balance — within each client')
    b.invert_yaxis(); b.grid(axis='x',alpha=.18); b.set_axisbelow(True)
    fig.suptitle('Federated training data across five clients',fontsize=18,weight='bold',y=.99)
    fig.text(.5,.91,f'Frozen seed-42 partition • {totals.sum():,} training samples • Before any forget request',ha='center',color='#555555')
    fig.text(.5,.025,'Left: absolute sample counts. Right: percentages normalized separately for each client.',ha='center',fontsize=10,color='#555555')
    fig.tight_layout(rect=[0,.06,1,.88])
    fig.savefig(out/'client_distribution.png',dpi=200,bbox_inches='tight')
    fig.savefig(out/'client_distribution.svg',bbox_inches='tight')
    rows=[{'client':i,'normal':int(counts[i,0]),'attack':int(counts[i,1]),'total':int(totals[i]),
           'attack_percent':float(percent[i,1]),'share_of_training_percent':float(100*totals[i]/totals.sum())} for i in x.tolist()]
    (out/'client_counts.json').write_text(json.dumps(rows,indent=2),encoding='utf-8')
    print(json.dumps(rows,indent=2))
    print(out/'client_distribution.png')


if __name__=='__main__':
    main()
