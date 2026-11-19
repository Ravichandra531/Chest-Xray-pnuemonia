#!/usr/bin/env python3
"""Evaluate frozen CNN embeddings, handcrafted descriptors and their fusion."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import balanced_accuracy_score
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.evaluation.metrics import compute_clinical_metrics
from src.models.cnn_pipeline import ManifestXrayDataset, image_transform, make_loader
from src.models.deep_learning import ChestXrayClassifier

class Features(ManifestXrayDataset):
    def __getitem__(self, index):
        tensor, label, path = super().__getitem__(index)
        with Image.open(self.root / path) as image:
            a = np.asarray(image.convert('L').resize((64, 64)), dtype=np.float32) / 255.0
        grid = a.reshape(8, 8, 8, 8).mean(axis=(1, 3)).ravel()
        hand = np.r_[grid, a.mean(), a.std(), np.abs(np.diff(a, axis=0)).mean(), np.abs(np.diff(a, axis=1)).mean()].astype(np.float32)
        return tensor, label, path, hand

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest', type=Path, default=ROOT/'data/processed/split_manifest.csv')
    p.add_argument('--checkpoint', type=Path, default=ROOT/'results/models/resnet18_best.pt')
    p.add_argument('--batch-size', type=int, default=16)
    p.add_argument('--output', type=Path, default=ROOT/'results/metrics/hybrid_ablation.json')
    args = p.parse_args()
    data = {}
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    payload = torch.load(args.checkpoint, map_location=device, weights_only=True)
    model = ChestXrayClassifier('resnet18', pretrained=False, freeze_backbone=True).to(device)
    model.load_state_dict(payload['state_dict']); model.eval()
    for split in ('train', 'val', 'test'):
        ds = Features(args.manifest, split, ROOT, image_transform(False))
        loader = make_loader(ds, args.batch_size, False, 42)
        embeddings, hand, labels, paths = [], [], [], []
        with torch.no_grad():
            for x, y, names, h in loader:
                embeddings.append(model.extract_embedding(x.to(device)).cpu().numpy())
                hand.append(h.numpy()); labels.extend(y.numpy().astype(int)); paths.extend(names)
        data[split] = (np.concatenate(embeddings), np.concatenate(hand), np.asarray(labels), np.asarray(paths))
        print(f'{split}: extracted {len(labels)} images', flush=True)
    results = {}
    predictions = pd.DataFrame({'path': data['test'][3], 'label': data['test'][2]})
    for name, indices in [('handcrafted', (1,)), ('cnn_embedding', (0,)), ('fusion', (0,1))]:
        def inputs(split): return np.concatenate([data[split][i] for i in indices], axis=1)
        best = None
        for c in (0.01, 0.1, 1.0):
            clf = make_pipeline(StandardScaler(), LogisticRegression(C=c, class_weight='balanced', max_iter=2000, random_state=42))
            clf.fit(inputs('train'), data['train'][2])
            score = balanced_accuracy_score(data['val'][2], clf.predict(inputs('val')))
            if best is None or score > best[0]: best = (score, c, clf)
        prob = best[2].predict_proba(inputs('test'))[:, 1]
        metrics = compute_clinical_metrics(data['test'][2], (prob >= .5).astype(int), prob)
        results[name] = {'selected_C': best[1], 'validation_balanced_accuracy': best[0], 'test': metrics.to_dict()}
        predictions[name + '_probability'] = prob
        print(name, json.dumps(results[name]), flush=True)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps({'description':'Exploratory reference-checkpoint feature ablation', 'threshold':0.5, 'models':results}, indent=2)+'\n')
    predictions.to_csv(args.output.with_name('hybrid_predictions.csv'), index=False)
if __name__ == '__main__': main()
