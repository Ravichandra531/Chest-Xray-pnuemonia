#!/usr/bin/env python3
"""Evaluate a trained ResNet18 once on the held-out test manifest."""
from __future__ import annotations
import argparse, json, sys
from pathlib import Path
import numpy as np, torch
ROOT=Path(__file__).resolve().parents[1]; sys.path.insert(0,str(ROOT))
from src.evaluation.metrics import compute_clinical_metrics
from src.models.cnn_pipeline import ManifestXrayDataset,image_transform,make_loader,predict_loader
from src.models.deep_learning import ChestXrayClassifier
def main(argv=None):
    p=argparse.ArgumentParser(); p.add_argument("--checkpoint",type=Path,default=ROOT/"results/models/resnet18_best.pt")
    p.add_argument("--manifest",type=Path,default=ROOT/"data/processed/split_manifest.csv"); p.add_argument("--batch-size",type=int,default=16)
    p.add_argument("--output",type=Path,default=ROOT/"results/metrics/resnet18_test_metrics.json"); args=p.parse_args(argv)
    if not args.checkpoint.is_file(): raise FileNotFoundError(f"CNN checkpoint missing: {args.checkpoint}")
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu"); payload=torch.load(args.checkpoint,map_location=device,weights_only=True)
    model=ChestXrayClassifier(payload.get("backbone","resnet18"),pretrained=False,freeze_backbone=False).to(device); model.load_state_dict(payload["state_dict"])
    data=ManifestXrayDataset(args.manifest,"test",ROOT,image_transform(False)); loader=make_loader(data,args.batch_size,False,42)
    y,prob,ids=predict_loader(model,loader,device); metrics=compute_clinical_metrics(y,(prob>=.5).astype(int),prob)
    result={"evidence_type":"real held-out test images","threshold":0.5,"sample_count":len(ids),"sample_ids":ids,"metrics":metrics.to_dict()}
    args.output.parent.mkdir(parents=True,exist_ok=True); args.output.write_text(json.dumps(result,indent=2)+"\n"); print(args.output); return 0
if __name__=="__main__": raise SystemExit(main())
