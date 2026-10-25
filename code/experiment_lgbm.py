"""Train/validation-only trial of boosted trees over frozen embeddings."""
import time
import joblib
import numpy as np
from lightgbm import LGBMClassifier, early_stopping, log_evaluation
from common import load_inputs
from features import make_features
from train import labels

tb,ta=load_inputs('train');vb,va=load_inputs('validation')
tx,tans=make_features(tb,ta);vx,vans=make_features(vb,va)
ty,vy=labels(tb,'train'),labels(vb,'validation')
hard=np.array([len(set(a for a in row if a is not None))>1 for row in tans])
vh=np.array([len(set(a for a in row if a is not None))>1 for row in vans])
te=ta['embeddings'];ve=va['embeddings']
tx=np.concatenate([tx,te,te-te.mean(axis=1,keepdims=True)],axis=2)
vx=np.concatenate([vx,ve,ve-ve.mean(axis=1,keepdims=True)],axis=2)
xt=tx[hard].reshape(-1,tx.shape[2]);yt=ty[hard].ravel()
xv=vx[vh].reshape(-1,vx.shape[2]);yv=vy[vh].ravel()
base=joblib.load('runs/model_bundle/model.joblib')
xx,_=make_features(vb,va)
p=base['hgb'].predict_proba(xx.reshape(-1,xx.shape[-1]))[:,1].reshape(-1,4)
bs=np.log(np.maximum(p,1e-5)/np.maximum(1-p,1e-5))
es=base['embedding_lr'].decision_function(base['scaler'].transform(ve.reshape(-1,1024))).reshape(-1,4)
bs+=base['embedding_weight']*es
def acc(s,mask):
    c=np.argmax(s,axis=1)
    return round(100*vy[np.arange(len(vy))[mask],c[mask]].mean(),3)
for leaf in (7,15,31):
 t=time.perf_counter()
 model=LGBMClassifier(n_estimators=400,num_leaves=leaf,max_depth=-1,learning_rate=.035,
     min_child_samples=40,reg_lambda=4.0,colsample_bytree=.7,subsample=.8,
     subsample_freq=1,verbosity=-1,n_jobs=4,random_state=7506)
 model.fit(xt,yt,eval_set=[(xv,yv)],callbacks=[early_stopping(25,verbose=False)])
 pr=model.predict_proba(vx.reshape(-1,vx.shape[2]))[:,1].reshape(-1,4)
 ls=np.log(np.maximum(pr,1e-5)/np.maximum(1-pr,1e-5))
 print('leaf',leaf,'trees',model.best_iteration_,'sec',round(time.perf_counter()-t,1),
       'alone',acc(ls,vh),'fusion',[(w,acc(bs+w*ls,vh)) for w in (.25,.5,1,2)],flush=True)
