"""Validation-selected residual ensemble. No references are used in inference."""
from pathlib import Path
import joblib
import numpy as np
from answer_ranker import build_groups
from boost_v8 import features
from neural_v6 import transform,logits

def predict(bank,arrays,checkpoint):
    a=joblib.load(Path(checkpoint)/'model.joblib')
    x,m,g=build_groups(bank,arrays,'numeric');s=x@a['linear']
    if a['tree_weight']:
        f=features(x,m)
        s+=a['tree_weight']*np.mean([md.predict(f.reshape(-1,f.shape[-1]),num_threads=4).reshape(m.shape) for md in a['trees']],axis=0)
    if a['neural_weight']:
        z=transform(x,m,a['mean'],a['scale'])
        s+=a['neural_weight']*np.mean([logits(z,md) for md in a['networks']],axis=0)
    s[~m]=-np.inf
    return [int(min(row[int(s[i].argmax())][1],key=lambda j:bank[i]['candidates'][j]['text'])) if row else 0 for i,row in enumerate(g)]
