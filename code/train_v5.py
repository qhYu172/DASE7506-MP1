"""Train-only PCA and subset pairs; validation selects dimensionality/regularization."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time
from itertools import combinations
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.decomposition import PCA
from common import load_inputs
from features import make_features
from answer_ranker import build_groups,encode_group,numeric_context
from train_answer_ranker import targets

def main():
    start=time.perf_counter();out=Path('runs/experiment_v5');out.mkdir(exist_ok=True)
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
    d=np.concatenate(diffs);w=np.asarray(pw);w/=w.sum()
    np.savez(out/'train_pairs.npz',d=d,w=w)
    pca=PCA(n_components=1024,svd_solver='full').fit(emb.reshape(-1,1024))
    sd=np.sqrt(np.average(d*d,axis=0,weights=w)).clip(1e-5)
    rec=[];best=None
    for dim in (32,64,128,256,512,1024):
      for whitening in (0.,.5,1.):
        projection=pca.components_[:dim].T/(pca.explained_variance_[:dim]**(whitening/2))
        dz=np.concatenate([d[:,:1024]@projection,d[:,1024:]/sd[1024:]],axis=1)
        # Use a common embedding scale, preserving the chosen whitening geometry.
        es=np.sqrt(np.average(dz[:,:dim]**2,axis=0,weights=w).mean());dz[:,:dim]/=es
        for alpha in (.001,.01,.1):
          def fun(coef):
            margin=dz@coef
            return w@np.logaddexp(0,-margin)+alpha*(coef@coef)/2,dz.T@(-w*expit(-margin))+alpha*coef
          fit=minimize(fun,np.zeros(dz.shape[1]),jac=True,method='L-BFGS-B',options={'maxiter':400,'ftol':1e-10})
          coeff=np.r_[projection@fit.x[:dim]/es,fit.x[dim:]/sd[1024:]]
          scores=vx@coeff;scores[~vm]=-np.inf;pred=scores.argmax(1)
          row=dict(dim=dim,whitening=whitening,alpha=alpha,validation_correct=int((pred==vy).sum()),converged=bool(fit.success))
          rec.append(row);print(json.dumps(row),flush=True)
          if best is None or row['validation_correct']>best['validation_correct']:
            best=row;np.savez(out/'ranker.npz',mean=np.zeros(len(coeff)),scale=np.ones(len(coeff)),weights=coeff,feature_mode='numeric')
          (out/'development.json').write_text(json.dumps(dict(selected=best,experiments=rec,training_seconds=time.perf_counter()-start,selection='validation only'),indent=2))
    print('SELECTED',best,flush=True)
if __name__=='__main__':main()
