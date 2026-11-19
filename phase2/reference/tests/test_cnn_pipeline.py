"""CPU smoke tests for the transfer-learning pipeline."""
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

from src.models.cnn_pipeline import ManifestXrayDataset, grad_cam, image_transform
from src.models.deep_learning import ChestXrayClassifier


def test_manifest_dataset_normalizes_pretrained_input(tmp_path: Path) -> None:
    image_path = tmp_path / "xray.png"
    Image.fromarray(np.full((32, 40), 127, dtype=np.uint8)).save(image_path)
    frame = pd.DataFrame([
        {"image_path": image_path.name, "label": "PNEUMONIA", "split": "train"}
    ])
    dataset = ManifestXrayDataset(frame, "train", tmp_path, image_transform(False, 64))
    tensor, label, sample_id = dataset[0]
    assert tensor.shape == (3, 64, 64)
    assert torch.isfinite(tensor).all()
    assert label.item() == 1.0
    assert sample_id == image_path.name


def test_resnet18_grad_cam_smoke() -> None:
    model = ChestXrayClassifier("resnet18", pretrained=False, freeze_backbone=True)
    model.eval()
    tensor = torch.randn(1, 3, 64, 64)
    heatmap = grad_cam(model, tensor)
    assert heatmap.shape == (64, 64)
    assert np.isfinite(heatmap).all()
    assert heatmap.min() >= 0.0 and heatmap.max() <= 1.0


def test_resnet18_grad_cam_targets_final_layer4_conv2() -> None:
    model = ChestXrayClassifier("resnet18", pretrained=False, freeze_backbone=True)
    assert model.get_target_layer_for_cam() is model.features[-1][-1].conv2
