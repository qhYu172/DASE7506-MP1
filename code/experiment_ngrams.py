"""Train/validation-only experiment: hashed text evidence on difficult groups."""
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.feature_extraction.text import HashingVectorizer, TfidfTransformer
from sklearn.linear_model import SGDClassifier

from common import load_inputs, parse_answer
from features import make_features
from train import labels

tb, ta = load_inputs('train')
vb, va = load_inputs('validation')
_, tans = make_features(tb, ta)
_, vans = make_features(vb, va)
ty, vy = labels(tb, 'train'), labels(vb, 'validation')
hard = np.array([len(set(a for a in aa if a is not None)) > 1 for aa in tans])
vh = np.array([len(set(a for a in aa if a is not None)) > 1 for aa in vans])
tt = [c['text'] for row, use in zip(tb, hard) if use for c in row['candidates']]
vt = [c['text'] for row in vb for c in row['candidates']]
base = joblib.load('runs/model_bundle/model.joblib')
vx,_=make_features(vb,va)
p=base['hgb'].predict_proba(vx.reshape(-1,vx.shape[-1]))[:,1].reshape(-1,4)
bs=np.log(np.maximum(p,1e-5)/np.maximum(1-p,1e-5))
es=base['embedding_lr'].decision_function(base['scaler'].transform(va['embeddings'].reshape(-1,1024))).reshape(-1,4)
bs += base['embedding_weight']*es

def acc(s, mask):
    pick=np.argmax(s,axis=1)
    return round(100*vy[np.arange(len(vy))[mask],pick[mask]].mean(),3)

for analyzer,ngrams,nfeat in [('word',(1,2),2**18),('char',(3,5),2**18)]:
    t=time.perf_counter()
    vec=HashingVectorizer(analyzer=analyzer,ngram_range=ngrams,n_features=nfeat,
                          alternate_sign=False,norm='l2',binary=True,dtype=np.float32)
    tr=vec.transform(tt); vr=vec.transform(vt)
    print('vectorized',analyzer,'seconds',round(time.perf_counter()-t,1),flush=True)
    for alpha in [1e-4,1e-5,1e-6]:
        model=SGDClassifier(loss='log_loss',alpha=alpha,penalty='l2',max_iter=150,
                            tol=1e-4,random_state=7506,average=True)
        model.fit(tr,ty[hard].ravel())
        ns=model.decision_function(vr).reshape(-1,4)
        print(analyzer,alpha,'alone',acc(ns,vh),'fusion',
              [(w,acc(bs+w*ns,vh)) for w in [.5,1,2,4]],flush=True)
