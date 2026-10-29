"""Listwise neural residuals, trained on train only; validation-only selection."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time
from itertools import combinations,product
from pathlib import Path
import joblib
import numpy as np
from scipy.special import softmax
from common import load_inputs
from features import make_features
from answer_ranker import build_groups,encode_group,numeric_context
from train_answer_ranker import targets
from neural_v6 import transform,logits

def main():
    start=time.perf_counter();out=Path('runs/experiment_v6');out.mkdir(exist_ok=True)
    tb,ta=load_inputs('train');vb,va=load_inputs('validation')
    x,mask,groups=build_groups(tb,ta,'numeric');y=targets(tb,groups,'train')
    vx,vm,vg=build_groups(vb,va,'numeric');vy=targets(vb,vg,'validation')
    surface,_=make_features(tb,ta);emb=ta['embeddings'].astype(float)
    xx=[];mm=[];yy=[];sw=[]
    for i in np.flatnonzero(mask.sum(1)>1):
        rr=[]
        for answer,members in groups[i]:
            numeric=numeric_context(tb[i]['question'],answer)
            rr.append([encode_group(emb[i],surface[i],list(sub),len(groups[i]),'numeric',numeric) for k in range(1,len(members)+1) for sub in combinations(members,k)])
        variants=list(product(*rr))
        for variant in variants:
            a=np.zeros_like(x[i]);a[:len(variant)]=variant
            xx.append(a);mm.append(mask[i]);yy.append(y[i]);sw.append(1/len(variants))
    x=np.array(xx);mask=np.array(mm);y=np.array(yy);sw=np.array(sw,dtype=np.float32);sw/=sw.mean()
    mean=x[mask].mean(0);scale=x[mask].std(0).clip(1e-5)
    z=transform(x,mask,mean,scale);vz=transform(vx,vm,mean,scale)
    with np.load('runs/model_bundle_v3/ranker.npz') as a:linear=a['weights']/a['scale']
    base=(x@linear).astype(np.float32);vbase=vx@linear
    records=[];best={'validation_correct':746};N=len(y)
    for hidden,reg in [(32,.003),(64,.003),(64,.03),(128,.03)]:
        ensemble={20:[],40:[],80:[]}
        for seed in (7506,7507,7508):
            rng=np.random.default_rng(seed)
            pars=[(rng.normal(size=(z.shape[2],hidden))*.02).astype(np.float32),np.zeros(hidden,np.float32),(rng.normal(size=hidden)*.01).astype(np.float32),np.zeros((),np.float32)]
            mom=[np.zeros_like(p) for p in pars];var=[np.zeros_like(p) for p in pars];step=0
            for epoch in range(1,81):
                order=rng.permutation(N)
                for starti in range(0,N,128):
                    ix=order[starti:starti+128];zz=z[ix];ww,b,v,c=pars
                    h=np.maximum(zz@ww+b,0);s=base[ix]+h@v+c;s[~mask[ix]]=-1e9
                    ds=softmax(s,axis=1);ds[np.arange(len(ix)),y[ix]]-=1;ds*=sw[ix,None]/len(ix)
                    dh=ds[:,:,None]*v*(h>0)
                    grads=[zz.reshape(-1,zz.shape[-1]).T@dh.reshape(-1,hidden)+reg*ww,dh.sum((0,1)),np.einsum('ngh,ng->h',h,ds)+reg*v,np.asarray(ds.sum())]
                    step+=1
                    for k,(p,g) in enumerate(zip(pars,grads)):
                        mom[k]=.9*mom[k]+.1*g;var[k]=.999*var[k]+.001*g*g
                        p-=.0003*(mom[k]/(1-.9**step))/(np.sqrt(var[k]/(1-.999**step))+1e-8)
                if epoch in ensemble:ensemble[epoch].append([p.copy() for p in pars])
        for epoch,models in ensemble.items():
            residual=np.mean([logits(vz,m) for m in models],axis=0)
            for weight in (.25,.5,1.):
                s=vbase+weight*residual;s[~vm]=-np.inf
                row=dict(hidden=hidden,reg=reg,epoch=epoch,weight=weight,validation_correct=int((s.argmax(1)==vy).sum()))
                records.append(row);print(json.dumps(row),flush=True)
                if row['validation_correct']>best['validation_correct']:
                    best=row.copy();joblib.dump(dict(models=models,mean=mean,scale=scale,linear=linear,weight=weight),out/'model.joblib')
            (out/'development.json').write_text(json.dumps(dict(selected=best,experiments=records,training_seconds=time.perf_counter()-start,selection='validation only; test not accessed',seeds=[7506,7507,7508]),indent=2))
    print('SELECTED',best,flush=True)
if __name__=='__main__':main()
