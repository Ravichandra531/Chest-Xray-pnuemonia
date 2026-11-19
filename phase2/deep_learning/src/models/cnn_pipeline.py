"""Reproducible transfer-learning data, training, inference, and Grad-CAM tools."""

from __future__ import annotations

import random
from dataclasses import dataclass
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch import nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms

from src.models.deep_learning import ChestXrayClassifier

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def seed_everything(seed: int = 42) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def image_transform(training: bool, image_size: int = 224) -> Callable:
    steps: list[Callable] = [transforms.Grayscale(num_output_channels=3), transforms.Resize((image_size, image_size))]
    if training:
        steps.extend([transforms.RandomHorizontalFlip(), transforms.RandomRotation(5)])
    steps.extend([transforms.ToTensor(), transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD)])
    return transforms.Compose(steps)


class ManifestXrayDataset(Dataset):
    """Manifest-backed dataset with strict path and label validation."""

    def __init__(self, manifest: str | Path | pd.DataFrame, split: str, project_root: str | Path,
                 transform: Callable | None = None):
        frame = pd.read_csv(manifest) if not isinstance(manifest, pd.DataFrame) else manifest.copy()
        required = {"image_path", "label", "split"}
        if not required.issubset(frame.columns):
            raise ValueError(f"Manifest requires columns {sorted(required)}")
        self.frame = frame.loc[frame["split"] == split].reset_index(drop=True)
        if self.frame.empty:
            raise ValueError(f"Manifest contains no rows for split={split!r}")
        invalid = set(self.frame["label"]) - {"NORMAL", "PNEUMONIA"}
        if invalid:
            raise ValueError(f"Unexpected labels: {sorted(invalid)}")
        self.root = Path(project_root)
        self.transform = transform or image_transform(training=split == "train")

    def __len__(self) -> int:
        return len(self.frame)

    def __getitem__(self, index: int):
        row = self.frame.iloc[index]
        path = self.root / str(row.image_path)
        if not path.is_file():
            raise FileNotFoundError(f"Manifest image is missing: {path}")
        with Image.open(path) as image:
            image = image.convert("L")
            tensor = self.transform(image)
        label = torch.tensor(float(row.label == "PNEUMONIA"), dtype=torch.float32)
        return tensor, label, str(row.image_path)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool, seed: int, num_workers: int = 0) -> DataLoader:
    generator = torch.Generator().manual_seed(seed)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, num_workers=num_workers,
                      generator=generator, pin_memory=torch.cuda.is_available())


@dataclass
class EpochRecord:
    epoch: int
    train_loss: float
    val_loss: float
    val_accuracy: float


def _run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, device: torch.device,
               optimizer: torch.optim.Optimizer | None = None) -> tuple[float, float]:
    model.train(optimizer is not None)
    total_loss = correct = count = 0
    for images, labels, _ in loader:
        images, labels = images.to(device), labels.to(device)
        if optimizer is not None:
            optimizer.zero_grad(set_to_none=True)
        with torch.set_grad_enabled(optimizer is not None):
            logits = model(images).squeeze(1)
            loss = criterion(logits, labels)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
        total_loss += float(loss.item()) * labels.numel()
        correct += int(((torch.sigmoid(logits) >= 0.5) == labels.bool()).sum().item())
        count += labels.numel()
    return total_loss / count, correct / count


def train_with_early_stopping(model: ChestXrayClassifier, train_loader: DataLoader, val_loader: DataLoader,
                              device: torch.device, epochs: int, learning_rate: float,
                              checkpoint_path: str | Path, patience: int = 3,
                              pos_weight: float | None = None) -> list[EpochRecord]:
    model.to(device)
    criterion = nn.BCEWithLogitsLoss(
        pos_weight=torch.tensor([pos_weight], device=device) if pos_weight is not None else None
    )
    optimizer = torch.optim.AdamW((p for p in model.parameters() if p.requires_grad), lr=learning_rate)
    best_loss, stale = float("inf"), 0
    history: list[EpochRecord] = []
    checkpoint_path = Path(checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    for epoch in range(1, epochs + 1):
        train_loss, _ = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_accuracy = _run_epoch(model, val_loader, criterion, device)
        history.append(EpochRecord(epoch, train_loss, val_loss, val_accuracy))
        if val_loss < best_loss:
            best_loss, stale = val_loss, 0
            torch.save({"state_dict": model.state_dict(), "backbone": model.backbone_name,
                        "epoch": epoch, "val_loss": val_loss}, checkpoint_path)
        else:
            stale += 1
            if stale >= patience:
                break
    model.load_state_dict(torch.load(checkpoint_path, map_location=device, weights_only=True)["state_dict"])
    return history


def predict_loader(model: nn.Module, loader: DataLoader, device: torch.device):
    model.eval()
    labels, probabilities, sample_ids = [], [], []
    with torch.no_grad():
        for images, batch_labels, batch_ids in loader:
            probs = torch.sigmoid(model(images.to(device)).squeeze(1)).cpu().numpy()
            labels.extend(batch_labels.numpy().astype(int).tolist())
            probabilities.extend(probs.tolist())
            sample_ids.extend(batch_ids)
    return np.asarray(labels), np.asarray(probabilities), sample_ids


def grad_cam(model: ChestXrayClassifier, input_tensor: torch.Tensor) -> np.ndarray:
    """Return Grad-CAM for the positive-class pneumonia logit.

    ``get_target_layer_for_cam`` resolves to ResNet18 layer4's final basic
    block ``conv2``. The returned map is attribution, not segmentation or a
    calibrated measure of anatomical importance.
    """
    if input_tensor.ndim != 4 or input_tensor.shape[0] != 1:
        raise ValueError("Grad-CAM expects a single BCHW input")
    # A frozen backbone has no trainable parameters, so its intermediate
    # activations do not require gradients unless the input does. Grad-CAM
    # still needs those activation gradients even though model weights remain
    # frozen and unchanged.
    input_tensor = input_tensor.detach().clone().requires_grad_(True)
    activations: list[torch.Tensor] = []
    gradients: list[torch.Tensor] = []
    layer = model.get_target_layer_for_cam()
    forward_handle = layer.register_forward_hook(lambda _m, _i, output: activations.append(output))
    backward_handle = layer.register_full_backward_hook(lambda _m, _gi, go: gradients.append(go[0]))
    try:
        model.zero_grad(set_to_none=True)
        pneumonia_logit = model(input_tensor).squeeze()
        if pneumonia_logit.numel() != 1:
            raise ValueError("Grad-CAM requires one binary pneumonia logit.")
        pneumonia_logit.backward()
        weights = gradients[0].mean(dim=(2, 3), keepdim=True)
        cam = torch.relu((weights * activations[0]).sum(dim=1, keepdim=True))
        cam = torch.nn.functional.interpolate(cam, input_tensor.shape[-2:], mode="bilinear", align_corners=False)
        cam = cam[0, 0]
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
        return cam.detach().cpu().numpy()
    finally:
        forward_handle.remove()
        backward_handle.remove()
