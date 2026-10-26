"""Validate important leakage invariants and serialized-model reproducibility."""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from src.pipeline import texture_features

root=Path(__file__).resolve().parent
manifest=pd.read_csv(root/'Datasets/processed/retained_manifest.csv')
dev=manifest.partition.eq('development').to_numpy()
assert not set(manifest.loc[dev,'group']) & set(manifest.loc[~dev,'group'])
assert not set(manifest.loc[dev,'pixel_hash']) & set(manifest.loc[~dev,'pixel_hash'])
assert manifest.pixel_hash.is_unique
folds=pd.read_csv(root/'results/cv_folds.csv')
assert folds.groupby('group').cv_fold.nunique().max()==1
cache=np.load(root/'Datasets/processed/features.npz')
assert cache['X'].shape==(len(manifest),1799)
assert np.isfinite(cache['X']).all()
assert np.array_equal(cache['y'],manifest.target)
artifact=joblib.load(root/'models/best_model.joblib')
raw=getattr(artifact['model'],artifact['score_method'])(cache['X'][~dev])
if raw.ndim==2: raw=raw[:,1]
saved=pd.read_csv(root/'results/test_predictions.csv')
assert np.allclose(raw,saved.score,atol=1e-7)
assert np.array_equal(raw>=artifact['threshold'],saved.prediction)
scaler=artifact['model'].named_steps['scale']
assert scaler.n_samples_seen_==int(dev.sum())
# Meaningful descriptor edge cases: constant image has no gradients and all-one LBP.
constant=np.ones((64,64),dtype=np.uint8)*100
h,l=texture_features(constant.astype(np.float32)/255,constant)
assert h.shape==(1764,) and np.allclose(h,0)
assert len(l)==10 and l[8]==1 and l.sum()==1
horizontal=np.tile(np.arange(64,dtype=np.float32),(64,1))
h,l=texture_features(horizontal/255,horizontal.astype(np.uint8))
assert np.isfinite(h).all() and h.sum()>0 and np.isclose(l.sum(),1)
print('PASS: partition/hash isolation, fold group isolation, feature dimensions/finiteness, saved-model score and prediction agreement, development-only final scaler, constant/gradient descriptor checks.')
