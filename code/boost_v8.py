"""Answer-group boosting blended with the frozen linear comparator."""
from pathlib import Path
import numpy as np
import joblib
from answer_ranker import build_groups

def features(x,mask):
    mean=(x*mask[:,:,None]).sum(1)/mask.sum(1).clip(1)[:,None]
    return np.concatenate([x,x[:,:,1024:]-mean[:,None,1024:]],axis=2)

def predict(bank,arrays,checkpoint):
    a=joblib.load(Path(checkpoint)/'model.joblib')
    x,m,g=build_groups(bank,arrays,'numeric');f=features(x,m)
    s=x@a['linear']
    p=np.mean([model.predict(f.reshape(-1,f.shape[-1])).reshape(m.shape) for model in a['models']],axis=0)
    s+=a['weight']*p;s[~m]=-np.inf
    return [int(min(row[int(s[i].argmax())][1],key=lambda j:bank[i]['candidates'][j]['text'])) if row else 0 for i,row in enumerate(g)]
