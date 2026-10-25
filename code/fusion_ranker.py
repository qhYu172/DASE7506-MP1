"""Probability-level fusion of our v2 selector and independently implemented group ranker."""
import json
from pathlib import Path
import numpy as np
from answer_ranker import build_groups
from submission import candidate_scores


def probabilities(scores,mask):
    scores=np.where(mask,scores,-1e30)
    p=np.exp(scores-scores.max(axis=1,keepdims=True))*mask
    return p/np.maximum(p.sum(axis=1,keepdims=True),1e-30)


def components(bank,arrays,base_path,ranker_path):
    cs,_,_=candidate_scores(bank,arrays,Path(base_path))
    with np.load(Path(ranker_path)/'ranker.npz',allow_pickle=False) as a:
        x,mask,groups=build_groups(bank,arrays,str(a['feature_mode']))
        group_scores=((x-a['mean'])/a['scale'])@a['weights']
    old=np.full(mask.shape,-1e30)
    for i,row in enumerate(groups):
        for j,(_,members) in enumerate(row):
            old[i,j]=cs[i,members].max()
    return probabilities(old,mask),probabilities(group_scores,mask),groups


def predict(bank,arrays,checkpoint):
    if not bank:return []
    checkpoint=Path(checkpoint)
    config=json.loads((checkpoint/'config.json').read_text())
    p,q,groups=components(bank,arrays,checkpoint/'v2',checkpoint/'group')
    scores=(1-config['group_weight'])*p+config['group_weight']*q
    result=[]
    for i,row in enumerate(groups):
        if not row:result.append(0);continue
        members=row[int(scores[i].argmax())][1]
        result.append(int(min(members,key=lambda j:bank[i]['candidates'][j]['text'])))
    return result
