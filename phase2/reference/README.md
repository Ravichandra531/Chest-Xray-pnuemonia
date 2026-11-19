# Chest X-Ray Classification with ResNet18 and Feature Fusion

This educational prototype classifies pediatric chest X-rays as NORMAL or PNEUMONIA. It is not a medical device and must not be used for patient decisions.

## What is implemented

The project classifies chest X-rays as NORMAL or PNEUMONIA using a ResNet18 convolutional neural network. Images are converted to grayscale, resized to 224 × 224 pixels, repeated across three channels, and normalized before inference. A supplied pretrained checkpoint generates a pneumonia score and a 512-dimensional representation of each image. The checkpoint was evaluated locally; it was not trained in this local run.

The hybrid implementation in `hybrid_ablation.py` combines the CNN representation with 68 handcrafted descriptors: 64 regional intensity averages, global mean and standard deviation, and horizontal and vertical intensity differences. A class-balanced logistic regression classifier evaluates handcrafted features alone, CNN features alone, and the combined features. This comparison tests whether combining the two feature types changes performance.

For each classifier, standardization is fitted on training data only. The regularization parameter C is selected from 0.01, 0.1, and 1.0 using validation balanced accuracy, and the test set is evaluated at a threshold of 0.5. The validation set also selected the neural-network checkpoint, so this is exploratory model development rather than an independent validation study.

## Dataset and splits

Download the [Chest X-Ray Images (Pneumonia) dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia). The manifest uses 4,432 training, 784 validation, and 624 test images. The manifest partitions the original train set and retained the original test set; original validation images are excluded. These splits differ from Phase 1's group-audited 618-image test set and the existing custom CNN report. Consequently, direct comparisons are not controlled. Exact duplicate checks do not prove patient-level independence; test data has previously been inspected. External validation is absent.

## Setup and execution

From the repository root, using Python 3.12:

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r phase2/reference/requirements.txt
mkdir -p phase2/reference/data/raw
ln -s /absolute/path/to/chest_xray phase2/reference/data/raw/chest_xray
python phase2/reference/scripts/evaluate_cnn.py --output phase2/reference/results/metrics/reproduced_test_metrics.json
python phase2/reference/scripts/hybrid_ablation.py
```

The linked folder must contain `train`, `val`, and `test`, with `NORMAL` and `PNEUMONIA` subfolders. On Windows, copy the extracted dataset into `phase2/reference/data/raw/chest_xray` instead of using a symbolic link. No raw images are committed. Scripts fail if images or checkpoint are missing. The fusion experiment requires feature extraction from all 5,840 images and can be slow on CPU.

To train a new model with the manifest split and default settings, use `python phase2/reference/scripts/train_cnn.py --checkpoint phase2/reference/results/models/new_run.pt`. This requires ImageNet weights to be available or downloaded. `--no-pretrained` changes the experiment. Training results must not be represented as identical to the supplied checkpoint without verification.

## CNN evaluation

The supplied checkpoint was evaluated locally using the dataset above. The local run produced TN=185, FP=49, FN=13, TP=377; accuracy 90.06%, sensitivity 96.67%, specificity 79.06%, ROC-AUC 0.9551. See `results/metrics/local_test_metrics.json`. These results evaluate the supplied checkpoint and do not represent a new local training run. Probabilities are model scores, not clinical certainty.

## Relationship to the existing report

`phase2/reports/Chest_Xray_Deep_Learning_Report.pdf` describes a different, five-block CNN and a 618-image test split. Its training scripts, trained weights and underlying per-image predictions were not present in the repository when this work began, so its results cannot be independently reproduced from this repository. Do not mix its scores with the ResNet18 scores or represent the two as a controlled comparison.

## Completed feature ablation

| Feature input | Validation balanced accuracy | Test balanced accuracy | Test sensitivity | Test specificity | Test ROC-AUC | TN / FP / FN / TP |
|---|---:|---:|---:|---:|---:|---|
| Handcrafted only | 94.79% | 68.63% | 97.95% | 39.32% | 0.9003 | 92 / 142 / 8 / 382 |
| CNN embedding only | 95.31% | 83.33% | 97.44% | 69.23% | 0.9368 | 162 / 72 / 10 / 380 |
| Fused features | 96.39% | 79.79% | 98.46% | 61.11% | 0.9473 | 143 / 91 / 6 / 384 |

Fusion improved test sensitivity and ROC-AUC compared with CNN embeddings alone, but reduced specificity and balanced accuracy. It missed four fewer pneumonia images and produced 19 more false positives. These results do not establish a general advantage for fusion. See the machine-readable `results/metrics/hybrid_ablation.json` and per-image `hybrid_predictions.csv`. Validation performance is much higher than test performance; investigate dataset distribution and possible leakage before drawing broader conclusions.
