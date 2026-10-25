"""Fit and compare MP2 rankers. Uses train references for fitting and validation for selection."""
import argparse
from collections import Counter
import json
from pathlib import Path
import time

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

from common import DATA, load_inputs, parse_answer, read_jsonl
from features import make_features


def labels(bank, split):
    gold = read_jsonl(DATA / f'gsm8k_{split}.jsonl')
    return np.array([[int(parse_answer(c['text']) == parse_answer(ref['answer']))
                      for c in row['candidates']] for row, ref in zip(bank, gold)], dtype=np.int8)


def choose(scores, answers, vote_weight=0.0):
    out = []
    for row_scores, row_answers in zip(scores, answers):
        counts = Counter(a for a in row_answers if a is not None)
        bonus = np.array([np.log(max(counts.get(a, 0), 1)) if a is not None else -1
                          for a in row_answers])
        out.append(int(np.argmax(row_scores + vote_weight * bonus)))
    return np.asarray(out)


def accuracy(scores, answers, y, vote_weight=0.0):
    c = choose(scores, answers, vote_weight)
    return 100 * y[np.arange(len(y)), c].mean()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--output', default='runs/model_bundle')
    args = ap.parse_args()
    t0 = time.perf_counter()
    train_bank, train_arrays = load_inputs('train')
    val_bank, val_arrays = load_inputs('validation')
    tx, ta = make_features(train_bank, train_arrays)
    vx, va = make_features(val_bank, val_arrays)
    ty, vy = labels(train_bank, 'train'), labels(val_bank, 'validation')
    hard = np.array([len(set(a for a in row if a is not None)) > 1 for row in ta])
    val_hard = np.array([len(set(a for a in row if a is not None)) > 1 for row in va])
    print('train hard', int(hard.sum()), 'validation hard', int(val_hard.sum()), flush=True)
    print('majority val', accuracy(np.zeros_like(vy, dtype=float), va, vy, 1.0), flush=True)

    te = train_arrays['embeddings'][hard].reshape(-1, 1024)
    ve = val_arrays['embeddings'].reshape(-1, 1024)
    scaler = StandardScaler().fit(te)
    elr = LogisticRegression(C=0.03, max_iter=300, random_state=7506)
    elr.fit(scaler.transform(te), ty[hard].ravel())
    es = elr.decision_function(scaler.transform(ve)).reshape(-1, 4)
    print('embedding val', accuracy(es, va, vy), 'hard', accuracy(es[val_hard], [va[i] for i in np.where(val_hard)[0]], vy[val_hard]), flush=True)

    models = []
    for depth in (3, 5):
        model = HistGradientBoostingClassifier(max_iter=160, max_leaf_nodes=2**depth,
                  learning_rate=0.05, l2_regularization=2.0, min_samples_leaf=35,
                  random_state=7506)
        tfit = time.perf_counter()
        model.fit(tx[hard].reshape(-1, tx.shape[-1]), ty[hard].ravel())
        hs = model.predict_proba(vx.reshape(-1, vx.shape[-1]))[:, 1].reshape(-1, 4)
        print('hgb', depth, 'fit_s', round(time.perf_counter()-tfit, 2),
              'val', accuracy(hs, va, vy), 'hard', accuracy(hs[val_hard], [va[i] for i in np.where(val_hard)[0]], vy[val_hard]), flush=True)
        models.append((model, hs))
    # Validation selection of a small, predeclared fusion grid.
    best = (-1, None)
    for idx, (_, hs) in enumerate(models):
        zhs = np.log(np.maximum(hs, 1e-5) / np.maximum(1-hs, 1e-5))
        for ew in (0, 0.15, 0.3, 0.5):
            for vw in (0, 0.5, 1.0, 1.5):
                s = zhs + ew * es
                acc = accuracy(s, va, vy, vw)
                if acc > best[0]:
                    best = (acc, (idx, ew, vw))
    print('best', best, 'elapsed', round(time.perf_counter()-t0, 2), flush=True)
    idx, ew, vw = best[1]
    out = Path(args.output)
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump({'hgb': models[idx][0], 'scaler': scaler, 'embedding_lr': elr,
                 'embedding_weight': ew, 'vote_weight': vw}, out/'model.joblib', compress=0)
    (out/'training.json').write_text(json.dumps({'seed': 7506, 'validation_accuracy': best[0],
        'selected_depth': (3, 5)[idx], 'embedding_weight': ew, 'vote_weight': vw,
        'train_hard_groups': int(hard.sum()), 'validation_hard_groups': int(val_hard.sum())}, indent=2))
    print('asset_bytes', sum(p.stat().st_size for p in out.iterdir()), flush=True)


if __name__ == '__main__':
    main()
