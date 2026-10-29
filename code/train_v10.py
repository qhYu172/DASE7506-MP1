"""Train a compact arithmetic correction, select on validation only."""
import os
for k in ('OMP_NUM_THREADS','OPENBLAS_NUM_THREADS','MKL_NUM_THREADS'):os.environ[k]='4'
import json,time,hashlib,datetime
from pathlib import Path
import numpy as np
from scipy.optimize import minimize
from scipy.special import expit
from common import load_inputs
from answer_ranker import build_groups
from train_answer_ranker import targets
from arithmetic_v10 import evidence

def main():
 start=time.perf_counter();out=Path('runs/experiment_v10');out.mkdir(exist_ok=True)
 tb,ta=load_inputs('train');vb,va=load_inputs('validation')
 x,m,g=build_groups(tb,ta,'numeric');y=targets(tb,g,'train');f=evidence(tb,g)
 vx,vm,vg=build_groups(vb,va,'numeric');vy=targets(vb,vg,'validation');vf=evidence(vb,vg)
 with np.load('runs/model_bundle_v3/ranker.npz') as a:linear=a['weights']/a['scale']
 base=x@linear;vbase=vx@linear;d=[];offset=[];pw=[]
 for i in np.flatnonzero(m.sum(1)>1):
  for j in np.flatnonzero(m[i]):
   if j!=y[i]:d.append(f[i,y[i]]-f[i,j]);offset.append(base[i,y[i]]-base[i,j]);pw.append(1/(m[i].sum()-1))
 d=np.asarray(d);offset=np.asarray(offset);pw=np.asarray(pw);pw/=pw.sum()
 scale=f[m].std(0).clip(1e-5);records=[];best={'validation_correct':746,'variant':'baseline','alpha':None,'strength':0}
 np.savez(out/'model.npz',linear=linear,correction=np.zeros(f.shape[-1]))
 for variant in ('format','arithmetic'):
  active=np.ones(48)
  if variant=='format':active=np.array([float(k%16<3) for k in range(48)])
  z=d/scale*active
  for alpha in (.01,.1,1.):
   def fun(w):
    margin=offset+z@w
    return pw@np.logaddexp(0,-margin)+alpha*(w@w)/2,z.T@(-pw*expit(-margin))+alpha*w
   fit=minimize(fun,np.zeros(48),method='L-BFGS-B',jac=True,options={'maxiter':500,'ftol':1e-11})
   for strength in (.25,.5,1.):
    correction=fit.x/scale*active*strength;s=vbase+vf@correction;s[~vm]=-np.inf
    row=dict(variant=variant,alpha=alpha,strength=strength,validation_correct=int((s.argmax(1)==vy).sum()),converged=bool(fit.success))
    records.append(row);print(json.dumps(row),flush=True)
    if row['validation_correct']>best['validation_correct']:
     best=row.copy();np.savez(out/'model.npz',linear=linear,correction=correction)
 (out/'development.json').write_text(json.dumps(dict(selected=best,experiments=records,training_seconds=time.perf_counter()-start,selection='validation only; references from test not opened'),indent=2))
 paths=['arithmetic_v10.py','train_v10.py','answer_ranker.py','features.py',str(out/'model.npz')]
 (out/'freeze.json').write_text(json.dumps(dict(frozen_at=datetime.datetime.now(datetime.timezone.utc).isoformat(),sha256={p:hashlib.sha256(Path(p).read_bytes()).hexdigest() for p in paths}),indent=2))
 print('SELECTED',best,flush=True)
if __name__=='__main__':main()
