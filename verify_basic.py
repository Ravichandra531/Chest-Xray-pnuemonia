"""Check the basic implementation and its genuinely executed notebook outputs."""
import ast
import csv
import json
from pathlib import Path
import numpy as np
from src.basic_ml import verify_math, predict_score, scale, metrics

ROOT = Path(__file__).resolve().parent
verify_math()
notebook = json.loads((ROOT/'Notebooks/Chest X-Rays Pneumonia.ipynb').read_text())
cells = [c for c in notebook['cells'] if c['cell_type'] == 'code']
assert all(c['execution_count'] is not None for c in cells)
assert not any(o['output_type'] == 'error' for c in cells for o in c['outputs'])
for cell in cells:
    tree = ast.parse(cell['source'])
    for node in ast.walk(tree):
        names = ([n.name for n in node.names] if isinstance(node, ast.Import)
                 else [node.module or ''] if isinstance(node, ast.ImportFrom) else [])
        assert not any(n.split('.')[0] in {'sklearn','pandas','scipy','tensorflow','torch','cv2','skimage'} for n in names)

out = ROOT/'basic_phase1'
cache = np.load(out/'basic_features.npz')
dev = cache['is_development']
assert cache['X'].shape[1] == 149 and np.isfinite(cache['X']).all()
assert not set(cache['groups'][dev]) & set(cache['groups'][~dev])
with (out/'development_folds.csv').open() as f:
    folds = list(csv.DictReader(f))
group_folds = {}
for row in folds:
    group_folds.setdefault(row['group'], set()).add(row['fold'])
assert all(len(fs) == 1 for fs in group_folds.values())
result = json.loads((out/'final_results.json').read_text())
saved = np.load(out/'model_parameters.npz')
model = {'kind': {'logistic_regression':'logistic','gaussian_naive_bayes':'naive_bayes','majority':'majority'}[result['selected_model']['model']]}
model.update({key:saved[key] for key in saved.files if not key.startswith('scaler_')})
scaler = {'mean':saved['scaler_mean'],'std':saved['scaler_std']}
assert np.allclose(scaler['mean'],cache['X'][dev].mean(axis=0))
score = predict_score(model,scale(cache['X'][~dev],scaler))
assert result['test_threshold'] == .5
computed = metrics(cache['y'][~dev],score)
assert all(np.isclose(value,result['test_metrics'][key]) for key,value in computed.items())
print(f'PASS: {len(cells)} executed cells; no ML-library imports; feature validity; disjoint groups/folds; development-fitted scaler; reloaded parameters reproduce test metrics at fixed threshold.')
