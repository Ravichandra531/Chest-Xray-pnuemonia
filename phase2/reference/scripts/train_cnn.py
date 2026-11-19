#!/usr/bin/env python3
"""Train a frozen-backbone ResNet18 with validation early stopping."""
from __future__ import annotations
import argparse, json, sys
from dataclasses import asdict
from pathlib import Path
import pandas as pd
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.models.cnn_pipeline import ManifestXrayDataset, image_transform, make_loader, seed_everything, train_with_early_stopping
from src.models.deep_learning import ChestXrayClassifier

def main(argv=None):
    p=argparse.ArgumentParser()
    p.add_argument("--manifest", type=Path, default=ROOT/"data/processed/split_manifest.csv")
    p.add_argument("--epochs", type=int, default=10); p.add_argument("--batch-size", type=int, default=16)
    p.add_argument("--lr", type=float, default=1e-3); p.add_argument("--patience", type=int, default=3)
    p.add_argument("--seed", type=int, default=42); p.add_argument("--no-pretrained", action="store_true")
    p.add_argument("--checkpoint", type=Path, default=ROOT/"results/models/resnet18_best.pt")
    args=p.parse_args(argv); seed_everything(args.seed)
    frame=pd.read_csv(args.manifest)
    missing=[str(ROOT/p) for p in frame.image_path if not (ROOT/p).is_file()]
    if missing: raise FileNotFoundError(f"Full dataset unavailable; {len(missing)} manifest images are missing. First: {missing[0]}")
    train=ManifestXrayDataset(frame,"train",ROOT,image_transform(True)); val=ManifestXrayDataset(frame,"val",ROOT,image_transform(False))
    train_loader=make_loader(train,args.batch_size,True,args.seed); val_loader=make_loader(val,args.batch_size,False,args.seed)
    counts=frame[frame.split=="train"].label.value_counts(); pos_weight=float(counts.get("NORMAL",0)/counts.get("PNEUMONIA",1))
    device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model=ChestXrayClassifier("resnet18",pretrained=not args.no_pretrained,freeze_backbone=True)
    history=train_with_early_stopping(model,train_loader,val_loader,device,args.epochs,args.lr,args.checkpoint,args.patience,pos_weight)
    history_path=args.checkpoint.with_suffix(".history.json")
    history_path.write_text(json.dumps({"seed":args.seed,"device":str(device),"pos_weight":pos_weight,"epochs":[asdict(x) for x in history]},indent=2)+"\n")
    print(f"Saved checkpoint: {args.checkpoint}\nSaved history: {history_path}")
    return 0
if __name__=="__main__": raise SystemExit(main())
