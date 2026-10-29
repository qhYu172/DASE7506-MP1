"""Validation-only blend of complementary logistic, margin and tree rankers."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json
from pathlib import Path
import joblib
import numpy as np
from common import load_inputs
from answer_ranker import build_groups
from train_answer_ranker import targets
from boost_v8 import features

out=Path('runs/experiment_v8_fusion');out.mkdir(exist_ok=True)
bundle=joblib.load('runs/experiment_v8/model.joblib')
with np.load('runs/experiment_v7/ranker.npz') as a:margin=a['weights']/a['scale']
tb,ta=load_inputs('train');tx,tm,_=build_groups(tb,ta,'numeric')
# Scale the margin model using only train-question score spreads.
def spread(w):
 s=tx@w;means=(s*tm).sum(1)/tm.sum(1)
 return np.sqrt(np.mean(((s-means[:,None])[tm])**2))
margin*=spread(bundle['linear'])/spread(margin)
vb,va=load_inputs('validation');x,m,g=build_groups(vb,va,'numeric');y=targets(vb,g,'validation')
f=features(x,m);trees=np.mean([md.predict(f.reshape(-1,f.shape[-1])).reshape(m.shape) for md in bundle['models']],axis=0)
best={'validation_correct':749};records=[]
for ratio in (0.,.25,.5,.75,1.):
 linear=(1-ratio)*bundle['linear']+ratio*margin
 for weight in (0.,.25,.5,.75,1.):
  s=x@linear+weight*trees;s[~m]=-np.inf
  row=dict(margin_weight=ratio,tree_weight=weight,validation_correct=int((s.argmax(1)==y).sum()));records.append(row)
  print(json.dumps(row),flush=True)
  if row['validation_correct']>best['validation_correct']:
   best=row.copy();joblib.dump(dict(models=bundle['models'],linear=linear,weight=weight),out/'model.joblib')
(out/'development.json').write_text(json.dumps(dict(selected=best,experiments=records,selection='validation only; spread normalization fitted on training'),indent=2))
print('SELECTED',best)
