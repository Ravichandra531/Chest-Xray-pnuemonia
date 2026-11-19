# Phase 2 presentation and viva guide

## Honest opening (30 seconds)

“Our Phase 2 investigates deep learning for binary pneumonia classification on pediatric chest X-rays. We reproduced an attributed reference ResNet18 model and evaluated a feature-fusion hybrid against its individual feature components. This is an educational research prototype, not a diagnostic tool.”

## Key questions

**What is transfer learning?** Initialize a model using knowledge learned on a different dataset, then adapt it to the target task. Here the reference used an ImageNet-initialized ResNet18 and trained a binary head. The reference checkpoint is supplied by the source project and its test evaluation was reproduced locally.

**What is the hybrid?** Extract a 512-value ResNet18 embedding and concatenate it with 68 interpretable image statistics. A class-balanced logistic regression learns a prediction from these combined features. The rationale is that learned spatial representations and coarse intensity/edge information may contain complementary signals. The ablation tests this hypothesis; combining features alone does not prove an improvement.

**How is leakage limited?** The reference manifest defines train, validation and test. Hybrid scalers and logistic models fit only on training data. Validation chooses regularization C. The source also used validation to choose the neural checkpoint, and the same test set has been examined before. This is exploratory, not a fresh independent confirmation. The reference's split does not guarantee patient separation.

**What does the ablation establish?** The handcrafted-only, CNN-only and fused logistic regressions hold data, classifier family, scaling, C selection and threshold constant while changing the features. It tests whether fusion adds value over both components in this experiment. A higher score does not prove a statistically significant gain or clinical validity.

**Explain sensitivity, specificity and balanced accuracy.** Sensitivity = TP/(TP+FN), specificity = TN/(TN+FP), and balanced accuracy is their mean. High sensitivity can coexist with many false positives. The reference ResNet18 evaluation has 377 TP, 13 FN, 185 TN and 49 FP among 624 test images.

**What is ROC-AUC?** A threshold-independent ranking measure across possible operating points. It does not establish probability calibration or performance at a chosen threshold.

**How was regularization used?** The reference applied training-only horizontal flips and small rotations, dropout in the classification head, weighted loss, AdamW, and validation-loss checkpoint selection with early stopping. The hybrid fits a standardized, regularized logistic regression, selecting C on validation.

**Why not compare the custom-CNN report directly?** Its reported split contains 618 test images, while the reference model uses 624. Its checkpoint and predictions were not present in the repository, so we cannot independently verify or reconstruct its reported results. Different splits prevent a fair direct comparison.

**What are the main limitations?** A single pediatric dataset, no verified patient-level separation in the reference split, no external evaluation, a previously inspected test set, and probability scores that are not validated for clinical use. Image-derived explanations such as Grad-CAM are not lesion segmentation or proof of clinical reasoning.

## Literature for discussion

- Kermany et al. (2018), “Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning,” *Cell*. https://doi.org/10.1016/j.cell.2018.02.010 — pediatric pneumonia and transfer learning in medical images.
- He et al. (2016), “Deep Residual Learning for Image Recognition,” CVPR. https://doi.org/10.1109/CVPR.2016.90 — residual connections enable deep network optimization.
- Rajpurkar et al. (2017), “CheXNet,” arXiv. https://arxiv.org/abs/1711.05225 — chest X-ray pneumonia classification using a deep convolutional model; it uses a different dataset and evaluation, so its reported performance is not directly comparable.
- Selvaraju et al. (2017), “Grad-CAM,” ICCV. https://doi.org/10.1109/ICCV.2017.74 — gradient-based localization for model interpretation.

## Demo sequence

1. Open `phase2/reference/README.md` and explain the origin of the checkpoint and split.
2. Show `results/metrics/local_test_metrics.json`, including sample count and confusion counts.
3. Show `results/metrics/hybrid_ablation.json` and compare all three feature inputs, if the run completed.
4. Explain limitations and answer questions using the points above.
