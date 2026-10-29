"""Question-fold CV refits anchor/scaler/correction; validation is a retention check."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time,hashlib,datetime
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from sklearn.model_selection import KFold
from common import load_inputs
from answer_ranker import build_groups
from train_answer_ranker import targets
from train_augmented_ranker import fit_pairs
from arithmetic_v10 import evidence
from repair_v11 import extra

def correction_fit(x,m,y,f,linear,indices,active,alpha):
    scale=f[indices][m[indices]].std(0).clip(1e-5);d=[];offset=[];pw=[];base=x@linear
    for i in indices:
        for j in np.flatnonzero(m[i]):
            if j!=y[i]:d.append(f[i,y[i]]-f[i,j]);offset.append(base[i,y[i]]-base[i,j]);pw.append(1/(m[i].sum()-1))
    z=np.asarray(d)/scale*active;offset=np.asarray(offset);pw=np.asarray(pw);pw/=pw.sum()
    def fun(w):
        s=offset+z@w
        return pw@np.logaddexp(0,-s)+alpha*(w@w)/2,z.T@(-pw*expit(-s))+alpha*w
    opt=minimize(fun,np.zeros(f.shape[-1]),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10})
    if not opt.success:raise RuntimeError(opt.message)
    return opt.x/scale*active

def main():
    start=time.perf_counter();out=Path('runs/experiment_v11');out.mkdir(exist_ok=True)
    bank,arr=load_inputs('train');x,m,g=build_groups(bank,arr,'numeric');y=targets(bank,g,'train')
    f=np.concatenate([evidence(bank,g),extra(bank,g)],axis=2);hard=np.flatnonzero(m.sum(1)>1)
    # Recover question boundaries of previously saved train-only subset pairs.
    cached=np.load('runs/experiment_v5/train_pairs.npz');d=cached['d'];w=cached['w']*len(hard);pairs=[];pw=[];offset=0
    for i in hard:
        lengths=[2**len(members)-1 for _,members in g[i]]
        count=sum(lengths[y[i]]*n for j,n in enumerate(lengths) if j!=y[i])
        pairs.append(d[offset:offset+count]);pw.append(w[offset:offset+count]);offset+=count
    assert offset==len(d)
    x,m,y,f=x[hard],m[hard],y[hard],f[hard]
    configs=[dict(variant='baseline',alpha=0.,strength=0.)]
    configs += [dict(variant=v,alpha=a,strength=s) for v in ('arithmetic','repair','combined') for a in (.01,.1,1.) for s in (.25,.5,1.)]
    correct=np.zeros(len(configs),int);folds=[]
    def active(v):return np.r_[np.full(48,v in ('arithmetic','combined')),np.full(46,v in ('repair','combined'))].astype(float)
    for fold,(tr,va) in enumerate(KFold(3,shuffle=True,random_state=7506).split(x)):
        fitted=fit_pairs(x[tr],m[tr],pairs,pw,tr,.01);linear=fitted[2]/fitted[1];cache={};fc=[]
        for k,c in enumerate(configs):
            if c['variant']=='baseline':coef=np.zeros(f.shape[-1])
            else:
                key=(c['variant'],c['alpha'])
                if key not in cache:cache[key]=correction_fit(x,m,y,f,linear,tr,active(c['variant']),c['alpha'])
                coef=cache[key]*c['strength']
            scores=x[va]@linear+f[va]@coef;scores[~m[va]]=-np.inf
            n=int((scores.argmax(1)==y[va]).sum());correct[k]+=n;fc.append(n)
        folds.append(fc);print('FOLD',fold,'baseline',fc[0],'best',max(fc),flush=True)
    results=[dict(**c,cv_correct=int(n),cv_n=len(y),fold_correct=[fold[k] for fold in folds]) for k,(c,n) in enumerate(zip(configs,correct))]
    selected=results[int(np.argmax(correct))]
    with np.load('runs/model_bundle_v3/ranker.npz') as a:linear=a['weights']/a['scale']
    c=selected
    coef=np.zeros(f.shape[-1]) if c['variant']=='baseline' else correction_fit(x,m,y,f,linear,np.arange(len(y)),active(c['variant']),c['alpha'])*c['strength']
    vb,va=load_inputs('validation');vx,vm,vg=build_groups(vb,va,'numeric');vy=targets(vb,vg,'validation');vf=np.concatenate([evidence(vb,vg),extra(vb,vg)],axis=2)
    scores=vx@linear+vf@coef;scores[~vm]=-np.inf
    validation_correct=int((scores.argmax(1)==vy).sum())
    np.savez(out/'model.npz',linear=linear,correction=coef)
    record=dict(selected=selected,experiments=results,validation_correct=validation_correct,validation_n=len(vy),training_seconds=time.perf_counter()-start,selection='3-fold train-question CV; all fold anchors/scalers/residuals refitted; validation retention check >=750',eligible_for_test=validation_correct>=750 and selected['cv_correct']>results[0]['cv_correct'])
    (out/'development.json').write_text(json.dumps(record,indent=2))
    paths=['repair_v11.py','train_v11.py','arithmetic_v10.py','answer_ranker.py','features.py',str(out/'model.npz')]
    (out/'freeze.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}),indent=2))
    print(json.dumps({k:v for k,v in record.items() if k!='experiments'}),flush=True)
if __name__=='__main__':main()
