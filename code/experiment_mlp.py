"""Train/validation-only nonlinear embedding experiment."""
import time
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from common import load_inputs
from features import make_features
from train import labels

tb,ta=load_inputs('train');vb,va=load_inputs('validation')
_,tans=make_features(tb,ta);_,vans=make_features(vb,va)
ty,vy=labels(tb,'train'),labels(vb,'validation')
hard=np.array([len(set(a for a in row if a is not None))>1 for row in tans])
vh=np.array([len(set(a for a in row if a is not None))>1 for row in vans])
te=ta['embeddings'][hard].reshape(-1,1024);ve=va['embeddings'].reshape(-1,1024)
sc=StandardScaler().fit(te);te=sc.transform(te);ve=sc.transform(ve)
base=joblib.load('runs/model_bundle/model.joblib')
xx,_=make_features(vb,va)
p=base['hgb'].predict_proba(xx.reshape(-1,xx.shape[-1]))[:,1].reshape(-1,4)
bs=np.log(np.maximum(p,1e-5)/np.maximum(1-p,1e-5))
es=base['embedding_lr'].decision_function(base['scaler'].transform(va['embeddings'].reshape(-1,1024))).reshape(-1,4)
bs+=base['embedding_weight']*es
def acc(s):
 c=np.argmax(s,axis=1);return round(100*vy[np.arange(len(vy))[vh],c[vh]].mean(),3)
for arch in [(32,),(64,),(128,),(64,32)]:
 for alpha in [.1,1.0]:
  t=time.perf_counter()
  m=MLPClassifier(hidden_layer_sizes=arch,alpha=alpha,batch_size=256,max_iter=150,
      early_stopping=True,validation_fraction=.15,n_iter_no_change=10,random_state=7506)
  m.fit(te,ty[hard].ravel())
  pr=m.predict_proba(ve)[:,1].reshape(-1,4)
  ms=np.log(np.maximum(pr,1e-5)/np.maximum(1-pr,1e-5))
  print(arch,alpha,'iter',m.n_iter_,'s',round(time.perf_counter()-t,1),'alone',acc(ms),
        'fusion',[(w,acc(bs+w*ms)) for w in (.25,.5,1)],flush=True)
