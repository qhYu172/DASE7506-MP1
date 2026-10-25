"""Frozen candidate ranker. No reference files are accessed at inference."""

from collections import Counter
import joblib
import numpy as np

from features import make_features


def candidate_scores(bank, arrays, checkpoint):
    """Return candidate logits, parsed answers, and the configured vote weight.

    bank: list of question dictionaries with four candidate texts each.
    arrays['embeddings']: float32 NumPy array of shape [N, 4, 1024].
    checkpoint: pathlib.Path or None, supplied through --checkpoint.
    """
    if checkpoint is None:
        raise ValueError('Pass --checkpoint runs/model_bundle')
    bundle = joblib.load(checkpoint / 'model.joblib')
    x, answers = make_features(bank, arrays)
    hgb = bundle['hgb'].predict_proba(x.reshape(-1, x.shape[-1]))[:, 1].reshape(-1, 4)
    scores = np.log(np.maximum(hgb, 1e-5) / np.maximum(1-hgb, 1e-5))
    ew = bundle['embedding_weight']
    if ew:
        embeddings = arrays['embeddings'].reshape(-1, 1024)
        emb_scores = bundle['embedding_lr'].decision_function(
            bundle['scaler'].transform(embeddings)).reshape(-1, 4)
        scores += ew * emb_scores
    if 'mlp' in bundle:
        embeddings = arrays['embeddings'].reshape(-1, 1024)
        probabilities = bundle['mlp'].predict_proba(
            bundle['mlp_scaler'].transform(embeddings))[:, 1].reshape(-1, 4)
        mlp_scores = np.log(np.maximum(probabilities, 1e-5) /
                            np.maximum(1-probabilities, 1e-5))
        scores += bundle['mlp_weight'] * mlp_scores
    return scores, answers, bundle['vote_weight']


def predict(bank, arrays, checkpoint):
    if checkpoint is not None and (checkpoint / 'ranker.npz').exists():
        from answer_ranker import predict as group_predict
        return group_predict(bank, arrays, checkpoint)
    scores, answers, vote_weight = candidate_scores(bank, arrays, checkpoint)
    result = []
    for row_scores, row_answers in zip(scores, answers):
        counts = Counter(a for a in row_answers if a is not None)
        bonus = np.array([np.log(max(counts.get(a, 0), 1)) if a is not None else -1
                          for a in row_answers])
        result.append(int(np.argmax(row_scores + vote_weight * bonus)))
    return result
