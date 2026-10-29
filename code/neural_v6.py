"""Listwise residual neural ranking with within-question context."""
from pathlib import Path
import joblib
import numpy as np
from answer_ranker import build_groups

def transform(x,mask,mean,scale):
    z=np.clip((x-mean)/scale,-8,8).astype(np.float32)
    center=(z*mask[:,:,None]).sum(1)/mask.sum(1).clip(1)[:,None]
    return np.concatenate([z,z-center[:,None,:]],axis=2)

def logits(z,model):
    w,b,v,c=model
    return np.maximum(z@w+b,0)@v+c

def predict(bank,arrays,checkpoint):
    asset=joblib.load(Path(checkpoint)/'model.joblib')
    x,mask,groups=build_groups(bank,arrays,'numeric')
    z=transform(x,mask,asset['mean'],asset['scale'])
    base=x@asset['linear']
    s=base+asset['weight']*np.mean([logits(z,m) for m in asset['models']],axis=0)
    s[~mask]=-np.inf
    return [int(min(g[int(s[i].argmax())][1],key=lambda j:bank[i]['candidates'][j]['text'])) if g else 0 for i,g in enumerate(groups)]
