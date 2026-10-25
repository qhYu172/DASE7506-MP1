"""Select group rankers using question-level train CV; report validation only after selection."""
import json
from pathlib import Path
import time
import numpy as np
from scipy.optimize import minimize
from scipy.special import logsumexp
from sklearn.model_selection import KFold
from common import DATA, load_inputs, parse_answer, read_jsonl
from answer_ranker import build_groups


def targets(bank, groups, split):
    refs = read_jsonl(DATA/f'gsm8k_{split}.jsonl')
    target = []
    for row, group, ref in zip(bank, groups, refs):
        assert row['id'] == ref['id'] and row['question'] == ref['question']
        answer = parse_answer(ref['answer'])
        target.append([a for a, _ in group].index(answer))
    return np.array(target)


def fit(x, mask, y, alpha):
    # Standardization sees only the fold's training questions; each question has total weight one.
    count = mask.sum(axis=1)
    sample_w = mask / count[:, None]
    mean = np.einsum('ng,ngd->d', sample_w, x) / len(x)
    var = np.einsum('ng,ngd->d', sample_w, np.square(x-mean)) / len(x)
    scale = np.sqrt(np.maximum(var, 1e-10))
    z = (x-mean)/scale
    # Center within each question to remove shared offsets and improve conditioning.
    z -= np.einsum('ng,ngd->nd', sample_w, z)[:,None,:]
    flat = z.reshape(-1, x.shape[-1])
    def objective(w):
        logits = (flat @ w).reshape(len(x),4)
        logits[~mask] = -1e30
        lse = logsumexp(logits, axis=1)
        loss = np.mean(lse-logits[np.arange(len(y)),y]) + alpha*np.dot(w,w)/2
        residual = np.exp(logits-lse[:,None])
        residual[np.arange(len(y)),y] -= 1
        gradient = flat.T @ residual.ravel()/len(y) + alpha*w
        return loss, gradient
    opt = minimize(objective, np.zeros(x.shape[-1]), jac=True, method='L-BFGS-B',
                   options={'maxiter':250, 'ftol':1e-9, 'gtol':1e-6})
    return mean, scale, opt.x, bool(opt.success), int(opt.nit)


def score(x, mask, params):
    mean, scale, weights = params[:3]
    s = ((x-mean)/scale) @ weights
    s[~mask] = -np.inf
    return s.argmax(axis=1)


def main():
    start=time.perf_counter()
    bank, arrays = load_inputs('train')
    records=[]; candidate_data={}
    for mode in ('embedding','pooled'):
        x, mask, groups = build_groups(bank, arrays, mode)
        y=targets(bank,groups,'train')
        hard=mask.sum(axis=1)>1
        x,mask,y=x[hard],mask[hard],y[hard]
        candidate_data[mode]=(x,mask,y)
        splits=list(KFold(n_splits=3,shuffle=True,random_state=7506).split(x))
        for alpha in (.001,.01,.1,1.0):
            oof=np.zeros(len(y),dtype=int)
            converged=[]
            for tr,va in splits:
                params=fit(x[tr],mask[tr],y[tr],alpha)
                oof[va]=score(x[va],mask[va],params)
                converged.append(params[3])
            row={'mode':mode,'alpha':alpha,'cv_correct':int((oof==y).sum()),
                 'cv_n':len(y),'cv_accuracy':float(100*(oof==y).mean()),
                 'all_folds_converged':all(converged)}
            records.append(row);print(json.dumps(row),flush=True)
    best=max(records,key=lambda r:r['cv_correct'])
    x,mask,y=candidate_data[best['mode']]
    params=fit(x,mask,y,best['alpha'])
    out=Path('runs/answer_ranker');out.mkdir(parents=True,exist_ok=True)
    np.savez(out/'ranker.npz',mean=params[0],scale=params[1],weights=params[2],feature_mode=best['mode'])
    vb,va=load_inputs('validation')
    vx,vm,vg=build_groups(vb,va,best['mode']);vy=targets(vb,vg,'validation')
    vp=score(vx,vm,params);vh=vm.sum(axis=1)>1
    results={'selection':'3-fold train-question CV; validation not used for model selection',
             'selected':best,'experiments':records,'seed':7506,
             'validation_correct':int((vp==vy).sum()),'validation_n':len(vy),
             'validation_hard_correct':int((vp[vh]==vy[vh]).sum()),'validation_hard_n':int(vh.sum()),
             'training_seconds':time.perf_counter()-start,'full_fit_converged':params[3]}
    (out/'development.json').write_text(json.dumps(results,indent=2),encoding='utf-8')
    print(json.dumps(results),flush=True)

if __name__=='__main__': main()
