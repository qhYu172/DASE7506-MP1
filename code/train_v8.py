"""Group-aware tree ensemble, trained only on train and selected on validation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time
from itertools import combinations,product
from pathlib import Path
import numpy as np
import joblib
from lightgbm import LGBMRanker
from common import load_inputs
from features import make_features
from answer_ranker import build_groups,encode_group,numeric_context
from train_answer_ranker import targets
from boost_v8 import features

def main():
 start=time.perf_counter();out=Path('runs/experiment_v8');out.mkdir(exist_ok=True)
 tb,ta=load_inputs('train');vb,va=load_inputs('validation')
 x,m,g=build_groups(tb,ta,'numeric');y=targets(tb,g,'train')
 vx,vm,vg=build_groups(vb,va,'numeric');vy=targets(vb,vg,'validation')
 sf,_=make_features(tb,ta);emb=ta['embeddings'].astype(float);xx=[];mm=[];yy=[];sw=[]
 for i in np.flatnonzero(m.sum(1)>1):
  rr=[]
  for answer,members in g[i]:
   numeric=numeric_context(tb[i]['question'],answer)
   rr.append([encode_group(emb[i],sf[i],list(sub),len(g[i]),'numeric',numeric) for k in range(1,len(members)+1) for sub in combinations(members,k)])
  variants=list(product(*rr))
  for variant in variants:
   a=np.zeros_like(x[i]);a[:len(variant)]=variant
   xx.append(a);mm.append(m[i]);yy.append(y[i]);sw.append(1/len(variants))
 x=np.array(xx);m=np.array(mm);y=np.array(yy);sw=np.array(sw);f=features(x,m);vf=features(vx,vm)
 target=np.zeros(m.shape,dtype=int);target[np.arange(len(y)),y]=1
 flat=f[m];labels=target[m];samplew=np.broadcast_to(sw[:,None],m.shape)[m]
 with np.load('runs/model_bundle_v3/ranker.npz') as a:linear=a['weights']/a['scale']
 baseline=vx@linear;best={'validation_correct':746};records=[]
 for leaves,l2 in [(7,10),(15,10),(15,50),(31,50)]:
  models=[]
  for seed in (7506,7507,7508):
   model=LGBMRanker(objective='lambdarank',n_estimators=300,learning_rate=.03,num_leaves=leaves,min_child_samples=60,reg_lambda=l2,colsample_bytree=.6,random_state=seed,n_jobs=4,verbosity=-1)
   model.fit(flat,labels,group=m.sum(1),sample_weight=samplew);models.append(model)
  for iterations in (50,100,200,300):
   ps=np.mean([model.predict(vf.reshape(-1,vf.shape[-1]),num_iteration=iterations).reshape(vm.shape) for model in models],axis=0)
   for weight in (.25,.5,1.,2.):
    s=baseline+weight*ps;s[~vm]=-np.inf
    row=dict(leaves=leaves,l2=l2,iterations=iterations,weight=weight,validation_correct=int((s.argmax(1)==vy).sum()))
    records.append(row);print(json.dumps(row),flush=True)
    if row['validation_correct']>best['validation_correct']:
     best=row.copy()
     # Store only the selected prefix of each trained tree ensemble.
     import lightgbm as lgb
     selected=[lgb.Booster(model_str=md.booster_.model_to_string(num_iteration=iterations)) for md in models]
     joblib.dump(dict(models=selected,linear=linear,weight=weight),out/'model.joblib')
   (out/'development.json').write_text(json.dumps(dict(selected=best,experiments=records,training_seconds=time.perf_counter()-start,seeds=[7506,7507,7508],selection='validation only'),indent=2))
 print('SELECTED',best,flush=True)
if __name__=='__main__':main()
