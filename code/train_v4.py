"""Train-only subset augmentation; validation chooses nonlinear architecture and blend."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time,warnings
from itertools import combinations
from pathlib import Path
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.exceptions import ConvergenceWarning
from common import load_inputs
from features import make_features
from answer_ranker import build_groups,encode_group,numeric_context
from train_answer_ranker import targets
from pairwise_v4 import pair_scores,linear_pair_scores

def main():
    warnings.filterwarnings('ignore',category=ConvergenceWarning)
    start=time.perf_counter();out=Path('runs/experiment_v4');out.mkdir(exist_ok=True)
    tb,ta=load_inputs('train');vb,va=load_inputs('validation')
    tx,tm,tg=build_groups(tb,ta,'numeric');ty=targets(tb,tg,'train')
    vx,vm,vg=build_groups(vb,va,'numeric');vy=targets(vb,vg,'validation')
    sf,_=make_features(tb,ta);emb=ta['embeddings'].astype(float)
    diffs=[];pw=[]
    for i in np.flatnonzero(tm.sum(1)>1):
        rr=[]
        for answer,members in tg[i]:
            numeric=numeric_context(tb[i]['question'],answer)
            rr.append(np.stack([encode_group(emb[i],sf[i],list(sub),len(tg[i]),'numeric',numeric) for k in range(1,len(members)+1) for sub in combinations(members,k)]))
        for j,neg in enumerate(rr):
            if j==ty[i]:continue
            d=(rr[ty[i]][:,None,:]-neg[None,:,:]).reshape(-1,tx.shape[-1])
            diffs.append(d);pw.extend([1/((len(rr)-1)*len(d))]*len(d))
    d=np.concatenate(diffs);w=np.asarray(pw);scale=np.sqrt(np.average(d*d,axis=0,weights=w)).clip(1e-5)
    z=np.clip(d/scale,-20,20).astype(np.float32);z=np.concatenate([z,-z]);y=np.r_[np.ones(len(d)),np.zeros(len(d))];w=np.r_[w,w];w/=w.mean()
    with np.load('runs/model_bundle_v3/ranker.npz') as asset:linear=tuple(asset[k].copy() for k in ('mean','scale','weights'))
    ls=linear_pair_scores(vx,vm,*linear);hard=vm.sum(1)>1
    records=[];best=None
    for hidden,alpha in [(32,1.),(64,1.),(64,10.),(128,10.)]:
        models=[];pred=[]
        for seed in (7506,7507,7508):
            m=MLPClassifier(hidden_layer_sizes=(hidden,),activation='tanh',alpha=alpha,batch_size=256,max_iter=100,learning_rate_init=.0005,random_state=seed,early_stopping=False,tol=1e-5,n_iter_no_change=15)
            m.fit(z,y,sample_weight=w);models.append(m);pred.append(pair_scores(vx,vm,m,scale))
        ns=np.mean(pred,axis=0)
        for weight in (0.,.25,.5,.75,1.):
            s=(1-weight)*ls+weight*ns;s[~vm]=-np.inf;p=s.argmax(1)
            row=dict(hidden=hidden,alpha=alpha,weight=weight,validation_correct=int((p==vy).sum()),hard_correct=int((p[hard]==vy[hard]).sum()),iterations=[m.n_iter_ for m in models])
            records.append(row);print(json.dumps(row),flush=True)
            if best is None or row['validation_correct']>best['validation_correct']:
                best=row.copy();joblib.dump(dict(models=models,scale=scale,linear=linear,weight=weight),out/'model.joblib',compress=0)
        (out/'development.json').write_text(json.dumps(dict(selected=best,experiments=records,training_seconds=time.perf_counter()-start,seeds=[7506,7507,7508],selection='validation only; no test data accessed'),indent=2))
    print('SELECTED',json.dumps(best),flush=True)
if __name__=='__main__':main()
