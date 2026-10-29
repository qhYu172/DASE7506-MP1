"""Confirm retrained anchor parameters match the shipped v3 prerequisite."""
import numpy as np
with np.load('runs/model_bundle_v3/ranker.npz') as a,np.load('runs/augmented_ranker/ranker.npz') as b:
    assert str(a['feature_mode'])==str(b['feature_mode'])
    for key in ('mean','scale','weights'):np.testing.assert_allclose(a[key],b[key],rtol=1e-7,atol=1e-9)
print('Rebuilt anchor matches the shipped training prerequisite.')
