import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from PIL import Image
from src.models.cnn_pipeline import ManifestXrayDataset, image_transform, grad_cam
from src.models.deep_learning import ChestXrayClassifier
from src.evaluation.metrics import compute_clinical_metrics


class Phase2Tests(unittest.TestCase):
    def test_data_transform_and_label(self):
        with tempfile.TemporaryDirectory() as folder:
            Image.fromarray(np.full((32, 40), 127, dtype=np.uint8)).save(
                Path(folder) / "image.png"
            )
            frame = pd.DataFrame(
                [{"image_path": "image.png", "label": "PNEUMONIA", "split": "test"}]
            )
            x, y, name = ManifestXrayDataset(
                frame, "test", folder, image_transform(False, 64)
            )[0]
            self.assertEqual(tuple(x.shape), (3, 64, 64))
            self.assertEqual(y.item(), 1)
            self.assertEqual(name, "image.png")

    def test_checkpoint_and_gradcam(self):
        root = Path(__file__).resolve().parents[1]
        checkpoint = torch.load(
            root / "results/models/resnet18_best.pt",
            map_location="cpu",
            weights_only=True,
        )
        model = ChestXrayClassifier("resnet18", pretrained=False).eval()
        model.load_state_dict(checkpoint["state_dict"])
        cam = grad_cam(model, torch.randn(1, 3, 64, 64))
        self.assertEqual(cam.shape, (64, 64))
        self.assertTrue(np.isfinite(cam).all())

    def test_metrics(self):
        y = np.array([0, 0, 1, 1])
        p = np.array([0.1, 0.7, 0.3, 0.9])
        m = compute_clinical_metrics(y, (p >= 0.5).astype(int), p)
        self.assertEqual(
            (m.true_negatives, m.false_positives, m.false_negatives, m.true_positives),
            (1, 1, 1, 1),
        )
        self.assertEqual(m.balanced_accuracy, 0.5)


if __name__ == "__main__":
    unittest.main()
