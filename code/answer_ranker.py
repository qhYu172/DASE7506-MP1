"""Independent answer-group ranking implementation; see PUBLIC_METHODS.md for ideas cited."""
from pathlib import Path
from fractions import Fraction
import re
import numpy as np
from common import parse_answer
from features import make_features


def numeric_context(question,answer):
    """Bounded numeric plausibility descriptors, not a solver or answer lookup."""
    numbers=[]
    for token in re.findall(r'(?<!\w)[+-]?\d+(?:,\d{3})*(?:\.\d+)?(?:/\d+)?',question):
        try:
            value=Fraction(token.replace(',',''))
            if abs(value)<=10**12:numbers.append(value)
        except (ValueError,ZeroDivisionError):pass
    a=Fraction(answer);magnitude=float(min(abs(a),10**12))
    maximum=max([float(abs(v)) for v in numbers]+[1.0])
    nearest=min([float(min(abs(a-v),10**24))/maximum for v in numbers]+[1e12])
    return np.array([a.denominator==1,np.log1p(magnitude/maximum),
        np.log1p(min(a.denominator,10**12)),a in numbers,
        np.log1p(len(numbers)),np.log1p(nearest)],dtype=float)


def encode_group(embedding, surface, members, group_count, feature_mode, numeric=None):
    block = embedding[members]
    avg = block.mean(axis=0)
    meta = np.array([len(members), np.log(len(members)),
                     np.square(block-avg).mean(), group_count], dtype=float)
    parts = [avg, meta]
    if feature_mode in ('pooled','numeric'):
        sx = surface[members, :39].copy()
        sx[:,28] = len(members)
        parts.extend([sx.mean(axis=0), sx.max(axis=0)-sx.min(axis=0)])
        if feature_mode=='numeric':parts.append(numeric)
    elif feature_mode != 'embedding':
        raise ValueError('Unknown feature mode')
    return np.concatenate(parts)


def build_groups(bank, arrays, feature_mode='pooled'):
    """Each valid final numeric answer receives one pooled feature vector."""
    surface, answers = make_features(bank, arrays)
    emb = np.asarray(arrays['embeddings'], dtype=np.float64)
    groups = []
    matrices = []
    for i, aa in enumerate(answers):
        members = {}
        for j, answer in enumerate(aa):
            if answer is not None:
                members.setdefault(answer, []).append(j)
        keys = sorted(members)
        vectors = []
        for answer in keys:
            ix = members[answer]
            numeric=numeric_context(bank[i]['question'],answer) if feature_mode=='numeric' else None
            vectors.append(encode_group(emb[i],surface[i],ix,len(keys),feature_mode,numeric))
        groups.append([(answer, members[answer]) for answer in keys])
        matrices.append(vectors)
    d = 1028 + (78 if feature_mode in ('pooled','numeric') else 0) + (6 if feature_mode=='numeric' else 0)
    x = np.zeros((len(bank), 4, d), dtype=np.float64)
    mask = np.zeros((len(bank), 4), dtype=bool)
    for i, vectors in enumerate(matrices):
        if vectors:
            x[i, :len(vectors)] = vectors
            mask[i, :len(vectors)] = True
    return x, mask, groups


def predict(bank, arrays, checkpoint):
    if not bank:
        return []
    with np.load(Path(checkpoint)/'ranker.npz', allow_pickle=False) as asset:
        mode = str(asset['feature_mode'])
        x, mask, groups = build_groups(bank, arrays, mode)
        scores = ((x-asset['mean'])/asset['scale']) @ asset['weights']
    scores[~mask] = -np.inf
    result = []
    for i, row in enumerate(groups):
        if not row:
            result.append(0)
            continue
        members = row[int(np.argmax(scores[i]))][1]
        # Numeric-answer scoring makes any member equivalent; text tie-break avoids index bias.
        chosen = min(members, key=lambda j: bank[i]['candidates'][j]['text'])
        result.append(int(chosen))
    return result
