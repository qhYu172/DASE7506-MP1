"""Train/validation-only group-context MLP experiment."""
import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
from common import load_inputs
from features import make_features
from train import labels

tb,ta=load_inputs('train');vb,va=load_inputs('validation')
tx,tans=make_features(tb,ta);vx,vans=make_features(vb,va)
ty,vy=labels(tb,'train'),labels(vb,'validation')
hard=np.array([len(set(a for a in row if a is not None))>1 for row in tans])
vh=np.array([len(set(a for a in row if a is not None))>1 for row in vans])
base=joblib.load('runs/model_bundle/model.joblib')
p=base['hgb'].predict_proba(vx.reshape(-1,vx.shape[-1]))[:,1].reshape(-1,4)
bs=np.log(np.maximum(p,1e-5)/np.maximum(1-p,1e-5))
es=base['embedding_lr'].decision_function(base['scaler'].transform(va['embeddings'].reshape(-1,1024))).reshape(-1,4)
bs+=base['embedding_weight']*es
def acc(s):
 c=np.argmax(s,axis=1);return round(100*vy[np.arange(len(vy))[vh],c[vh]].mean(),3)
for kind in ('relative','both','both_features'):
 te=ta['embeddings'];ve=va['embeddings']
 rt=te-te.mean(axis=1,keepdims=True);rv=ve-ve.mean(axis=1,keepdims=True)
 if kind=='relative': xt=rt; xv=rv
 elif kind=='both': xt=np.concatenate([te,rt],axis=2);xv=np.concatenate([ve,rv],axis=2)
 else: xt=np.concatenate([te,rt,tx],axis=2);xv=np.concatenate([ve,rv,vx],axis=2)
 xt=xt[hard].reshape(-1,xt.shape[2]);xv=xv.reshape(-1,xv.shape[2])
 sc=StandardScaler().fit(xt);xt=sc.transform(xt);xv=sc.transform(xv)
 for alpha in (0.1,1.0):
  m=MLPClassifier(hidden_layer_sizes=(64,),alpha=alpha,batch_size=256,max_iter=150,
      early_stopping=True,validation_fraction=.15,n_iter_no_change=10,random_state=7506)
  m.fit(xt,ty[hard].ravel())
  pr=m.predict_proba(xv)[:,1].reshape(-1,4)
  ms=np.log(np.maximum(pr,1e-5)/np.maximum(1-pr,1e-5))
  print(kind,alpha,'iter',m.n_iter_,'alone',acc(ms),
        'fusion',[(w,acc(bs+w*ms)) for w in (.25,.5,1,1.5)],flush=True)
