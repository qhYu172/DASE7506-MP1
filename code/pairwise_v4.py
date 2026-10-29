"""Nonlinear antisymmetric answer comparison; inference never reads references."""
from pathlib import Path
import joblib
import numpy as np
from scipy.special import expit
from answer_ranker import build_groups

def pair_scores(x, mask, model, scale):
    out = np.zeros(mask.shape)
    for a in range(4):
        for b in range(a+1,4):
            use = mask[:,a] & mask[:,b]
            if not use.any(): continue
            delta = np.clip((x[use,a]-x[use,b])/scale,-20,20).astype(np.float32)
            # Antisymmetrize probabilities so reversing a pair reverses its vote.
            p = .5*(model.predict_proba(delta)[:,1]+1-model.predict_proba(-delta)[:,1])
            out[use,a] += p
            out[use,b] += 1-p
    return out

def linear_pair_scores(x, mask, mean, scale, weights):
    logits = ((x-mean)/scale)@weights
    out = np.zeros(mask.shape)
    for a in range(4):
        for b in range(a+1,4):
            use=mask[:,a]&mask[:,b]
            p=expit(logits[use,a]-logits[use,b])
            out[use,a]+=p;out[use,b]+=1-p
    return out

def predict(bank, arrays, checkpoint):
    bundle=joblib.load(Path(checkpoint)/'model.joblib')
    x,mask,groups=build_groups(bank,arrays,'numeric')
    scores=linear_pair_scores(x,mask,*bundle['linear'])
    if bundle['weight']:
        nonlinear=sum(pair_scores(x,mask,m,bundle['scale']) for m in bundle['models'])/len(bundle['models'])
        scores=(1-bundle['weight'])*scores+bundle['weight']*nonlinear
    scores[~mask]=-np.inf
    return [int(min(g[int(scores[i].argmax())][1],key=lambda j:bank[i]['candidates'][j]['text'])) if g else 0 for i,g in enumerate(groups)]
