"""Compare robust linear pair losses on training subsets; select on validation."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from common import load_inputs
from answer_ranker import build_groups
from train_answer_ranker import targets

def main():
 start=time.perf_counter();out=Path('runs/experiment_v7');out.mkdir(exist_ok=True)
 with np.load('runs/experiment_v5/train_pairs.npz') as a:d=a['d'];pw=a['w']
 with np.load('runs/model_bundle_v3/ranker.npz') as a:scale=a['scale'];init=a['weights']
 bank,arr=load_inputs('validation');x,mask,g=build_groups(bank,arr,'numeric');y=targets(bank,g,'validation')
 z=d/scale;records=[];best={'validation_correct':746}
 for loss in ('logistic','squared_hinge','huber_hinge'):
  for alpha in (.0003,.001,.003,.01,.03,.1,.3):
   def fun(coef):
    margin=z@coef
    if loss=='logistic':l=np.logaddexp(0,-margin);grad=-expit(-margin)
    else:
     err=np.maximum(1-margin,0)
     if loss=='squared_hinge':l=err**2/2;grad=-err
     else:l=np.where(err<1,err**2/2,err-.5);grad=-np.minimum(err,1)
    return pw@l+alpha*(coef@coef)/2,z.T@(pw*grad)+alpha*coef
   opt=minimize(fun,init.copy(),jac=True,method='L-BFGS-B',options={'maxiter':500,'ftol':1e-10})
   scores=(x/scale)@opt.x;scores[~mask]=-np.inf
   row=dict(loss=loss,alpha=alpha,validation_correct=int((scores.argmax(1)==y).sum()),converged=bool(opt.success));records.append(row);print(json.dumps(row),flush=True)
   if row['validation_correct']>best['validation_correct']:
    best=row.copy();np.savez(out/'ranker.npz',mean=np.zeros(len(scale)),scale=scale,weights=opt.x,feature_mode='numeric')
   (out/'development.json').write_text(json.dumps(dict(selected=best,experiments=records,training_seconds=time.perf_counter()-start,selection='validation only'),indent=2))
 print('SELECTED',best,flush=True)
if __name__=='__main__':main()
