"""Train-only question CV for subset-augmented pairwise answer-group ranking.

General subset-augmentation idea credited to CaspianW in PUBLIC_METHODS.md.
This is an independent implementation using this project's existing features.
"""
from itertools import combinations
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import KFold
from common import load_inputs
from features import make_features
from answer_ranker import build_groups, encode_group, numeric_context
from train_answer_ranker import targets, score


def fit_pairs(x,mask,pairs,pair_weights,indices,alpha):
    sample_w=mask/mask.sum(axis=1)[:,None]
    mean=np.einsum('ng,ngd->d',sample_w,x)/len(x)
    var=np.einsum('ng,ngd->d',sample_w,np.square(x-mean))/len(x)
    scale=np.sqrt(np.maximum(var,1e-10))
    delta=np.concatenate([pairs[i] for i in indices])/scale
    weights=np.concatenate([pair_weights[i] for i in indices])/len(indices)
    def objective(w):
        margin=delta@w
        loss=np.dot(weights,np.logaddexp(0,-margin))+alpha*np.dot(w,w)/2
        grad=delta.T@(-weights*expit(-margin))+alpha*w
        return loss,grad
    opt=minimize(objective,np.zeros(x.shape[-1]),method='L-BFGS-B',jac=True,
                 options={'maxiter':300,'ftol':1e-9,'gtol':1e-6})
    return mean,scale,opt.x,bool(opt.success),int(opt.nit)


def main():
    start=time.perf_counter();bank,arrays=load_inputs('train')
    surface,_=make_features(bank,arrays);emb=arrays['embeddings'].astype(float)
    records=[];datasets={}
    for mode in ('embedding','pooled','numeric'):
        x,mask,groups=build_groups(bank,arrays,mode);y=targets(bank,groups,'train')
        hard=np.flatnonzero(mask.sum(axis=1)>1)
        pairs=[];pair_weights=[]
        for i in hard:
            representations=[]
            for answer,members in groups[i]:
                subsets=[list(sub) for k in range(1,len(members)+1) for sub in combinations(members,k)]
                numeric=numeric_context(bank[i]['question'],answer) if mode=='numeric' else None
                representations.append(np.stack([encode_group(emb[i],surface[i],sub,len(groups[i]),mode,numeric) for sub in subsets]))
            positive=representations[y[i]];diffs=[];weights=[]
            for j,negative in enumerate(representations):
                if j==y[i]:continue
                diff=(positive[:,None,:]-negative[None,:,:]).reshape(-1,x.shape[-1])
                diffs.append(diff);weights.extend([1/((len(groups[i])-1)*len(diff))]*len(diff))
            pairs.append(np.concatenate(diffs));pair_weights.append(np.array(weights))
        x,mask,y=x[hard],mask[hard],y[hard]
        datasets[mode]=(x,mask,y,pairs,pair_weights)
        for alpha in (.001,.01,.1):
            oof=np.zeros(len(y),dtype=int);converged=[]
            for tr,va in KFold(n_splits=3,shuffle=True,random_state=7506).split(x):
                params=fit_pairs(x[tr],mask[tr],pairs,pair_weights,tr,alpha)
                oof[va]=score(x[va],mask[va],params);converged.append(params[3])
            row={'mode':mode,'alpha':alpha,'cv_correct':int((oof==y).sum()),'cv_n':len(y),
                 'cv_accuracy':float(100*(oof==y).mean()),'all_folds_converged':all(converged)}
            records.append(row);print(json.dumps(row),flush=True)
    best=max(records,key=lambda r:r['cv_correct']);x,mask,y,pairs,pw=datasets[best['mode']]
    params=fit_pairs(x,mask,pairs,pw,np.arange(len(y)),best['alpha'])
    out=Path('runs/augmented_ranker');out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'ranker.npz',mean=params[0],scale=params[1],weights=params[2],feature_mode=best['mode'])
    vb,va=load_inputs('validation');vx,vm,vg=build_groups(vb,va,best['mode'])
    vy=targets(vb,vg,'validation');vp=score(vx,vm,params);vh=vm.sum(axis=1)>1
    results={'selection':'3-fold train-question CV','selected':best,'experiments':records,'seed':7506,
             'validation_correct':int((vp==vy).sum()),'validation_n':len(vy),
             'validation_hard_correct':int((vp[vh]==vy[vh]).sum()),'validation_hard_n':int(vh.sum()),
             'training_seconds':time.perf_counter()-start,'full_fit_converged':params[3]}
    (out/'development.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results),flush=True)

if __name__=='__main__':main()
