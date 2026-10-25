"""Freeze the validation-selected nonlinear embedding addition to the MP2 selector."""
import json
from pathlib import Path

import joblib
import numpy as np
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler

from common import load_inputs
from features import make_features
from train import labels


def main():
    bank, arrays = load_inputs('train')
    _, answers = make_features(bank, arrays)
    y = labels(bank, 'train')
    hard = np.array([len(set(a for a in row if a is not None)) > 1 for row in answers])
    embeddings = arrays['embeddings'][hard].reshape(-1, 1024)
    scaler = StandardScaler().fit(embeddings)
    model = MLPClassifier(hidden_layer_sizes=(64,), alpha=1.0, batch_size=256,
                          max_iter=150, early_stopping=True, validation_fraction=.15,
                          n_iter_no_change=10, random_state=7506)
    model.fit(scaler.transform(embeddings), y[hard].ravel())
    bundle = joblib.load('runs/model_bundle/model.joblib')
    bundle['mlp'] = model
    bundle['mlp_scaler'] = scaler
    bundle['mlp_weight'] = 1.0
    out = Path('runs/model_bundle_v2')
    out.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, out/'model.joblib', compress=0)
    (out/'training.json').write_text(json.dumps({
        'seed': 7506, 'base': 'model_bundle', 'train_hard_groups': int(hard.sum()),
        'mlp_hidden_layers': [64], 'mlp_alpha': 1.0, 'mlp_weight': 1.0,
        'selection_metric': 'validation hard-group accuracy',
        'validation_hard_accuracy': 83.66197183098592,
    }, indent=2))
    print('mlp_iter', model.n_iter_, 'asset_bytes', sum(p.stat().st_size for p in out.iterdir()))


if __name__ == '__main__':
    main()
