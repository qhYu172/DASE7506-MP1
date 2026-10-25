"""Validation-only diagnostics; never called by the frozen predictor."""
from collections import defaultdict
import joblib
import numpy as np

from common import load_inputs
from features import make_features
from train import labels


bank, arrays = load_inputs('validation')
x, answers = make_features(bank, arrays)
y = labels(bank, 'validation')
b = joblib.load('runs/model_bundle/model.joblib')
p = b['hgb'].predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(-1, 4)
z = np.log(np.maximum(p, 1e-5) / np.maximum(1-p, 1e-5))
e = b['embedding_lr'].decision_function(b['scaler'].transform(arrays['embeddings'].reshape(-1, 1024))).reshape(-1, 4)
s = z + b['embedding_weight'] * e
for weight in (0, 0.25, 0.5, 1.0, 2.0):
    picks=[]
    for sc, aa in zip(s, answers):
        grouped=defaultdict(list)
        for j,a in enumerate(aa): grouped[a].append(j)
        gs={a: max(sc[ids]) + weight*np.log(len(ids)) for a,ids in grouped.items()}
        winning=max(gs,key=gs.get)
        picks.append(max(grouped[winning],key=lambda j:sc[j]))
    print('group',weight,round(100*y[np.arange(len(y)),picks].mean(),3))
print('selected', round(100*y[np.arange(len(y)),np.argmax(s,axis=1)].mean(),3))
chosen=np.argmax(s,axis=1)
for m in (1,2,3,4):
    subset=np.array([max(__import__('collections').Counter(a for a in row if a is not None).values())==m for row in answers])
    if subset.any(): print('max_vote',m,'n',int(subset.sum()),'accuracy',round(100*y[np.arange(len(y))[subset],chosen[subset]].mean(),3))
hard=np.array([len(set(a for a in row if a is not None))>1 for row in answers])
print('hard_n',int(hard.sum()),'hard_accuracy',round(100*y[np.arange(len(y))[hard],chosen[hard]].mean(),3))
print('correct_choices',int(y[np.arange(len(y)),chosen].sum()))
maj=[]
for aa in answers:
    counts=__import__('collections').Counter(a for a in aa if a is not None)
    maj.append(max(range(4),key=lambda j:counts.get(aa[j],0)))
print('majority_hard_accuracy',round(100*y[np.arange(len(y))[hard],np.array(maj)[hard]].mean(),3))
margin=np.sort(s,axis=1)[:,-1]-np.sort(s,axis=1)[:,-2]
edges=np.quantile(margin[hard],[0,.25,.5,.75,1])
for k in range(4):
    ix=np.where(hard)[0][np.argsort(margin[hard])[k*int(hard.sum())//4:(k+1)*int(hard.sum())//4 if k<3 else int(hard.sum())]]
    print('hard_margin_quartile',k+1,'n',len(ix),'accuracy',round(100*y[ix,chosen[ix]].mean(),3))
for i in np.where(y[np.arange(len(y)),chosen]==0)[0][:3]:
    print('error_example',i,'question',bank[i]['question'][:220],
          'answers',answers[i],'selected',int(chosen[i]),'correct',np.where(y[i])[0].tolist())
