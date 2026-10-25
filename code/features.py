"""Deterministic, reference-free features for MP2 candidate ranking."""
from collections import Counter
from fractions import Fraction
import re

import numpy as np

from common import parse_answer

NUMBER = re.compile(r"[-+]?\d[\d,]*(?:\.\d+)?(?:/\d+)?")
WORD = re.compile(r"[a-z]+")


def _number_set(text):
    out = set()
    for match in NUMBER.findall(text):
        try:
            out.add(str(Fraction(match.replace(',', ''))))
        except (ValueError, ZeroDivisionError):
            pass
    return out


def make_features(bank, arrays):
    """Return [N,4,F] numeric features and parsed answer groups."""
    n = len(bank)
    emb = np.asarray(arrays['embeddings'], dtype=np.float32)
    norm = emb / np.maximum(np.linalg.norm(emb, axis=2, keepdims=True), 1e-8)
    centre = norm.mean(axis=1, keepdims=True)
    sim = np.matmul(norm, norm.transpose(0, 2, 1))
    answers = [[parse_answer(c['text']) for c in row['candidates']] for row in bank]
    rows = []
    for i, row in enumerate(bank):
        question = row['question']
        qnums = _number_set(question)
        qwords = set(WORD.findall(question.lower()))
        count = Counter(a for a in answers[i] if a is not None)
        group = []
        for j, cand in enumerate(row['candidates']):
            s = cand['text']; lo = s.lower()
            nums = _number_set(s)
            words = WORD.findall(lo)
            aw = answers[i][j]
            other_sims = np.delete(sim[i, j], j)
            try:
                aval = float(Fraction(aw)) if aw is not None else 0.0
            except (ValueError, OverflowError):
                aval = 0.0
            last = s[-180:].lower()
            feat = [
                len(s), len(words), len(s.splitlines()), len(NUMBER.findall(s)),
                len(nums), len(qnums & nums), len(qnums - nums),
                len(set(words) & qwords) / max(len(qwords), 1),
                len(set(words)), len(s) / max(len(question), 1),
                s.count('='), s.count('+'), s.count('-'), s.count('*'),
                s.count('/'), s.count('\\frac'), s.count('\\boxed'),
                s.count('$'), s.count('**'), s.count('####'),
                int('answer' in last), int('therefore' in last),
                int('check' in lo), int('verify' in lo),
                int('however' in lo), int('alternatively' in lo),
                int('sorry' in lo or 'mistake' in lo or 'correcting' in lo),
                int(aw is not None), count.get(aw, 0) if aw is not None else 0,
                len(count), int(aw in qnums), np.log1p(abs(aval)),
                int(aval < 0), int(aval == 0),
                float(np.dot(norm[i, j], centre[i, 0])),
                float(other_sims.mean()), float(other_sims.max()),
                float(other_sims.min()), float(np.linalg.norm(emb[i, j])),
            ]
            group.append(feat)
        rows.append(group)
    x = np.asarray(rows, dtype=np.float32)
    # Candidate-relative values expose unusual length and formatting within a question.
    relative = x - x.mean(axis=1, keepdims=True)
    return np.concatenate([x, relative], axis=2), answers
