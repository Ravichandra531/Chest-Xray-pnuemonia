# Chest X-Ray Pneumonia Classification: Phase 1 Technical Report

## Objective and scope

Evaluate classical machine learning on the NORMAL/PNEUMONIA labels of the downloaded Kaggle chest X-ray dataset. The educational objective is a defensible, reproducible comparison, including failures and limitations. The system does not provide clinical diagnoses. Neural networks and deployment are outside Phase 1.

The organization follows the reference [inventory forecasting repository](https://github.com/tejuuu-7774/Inventory-demand-forecasting): exploration, feature preparation, model development, saved predictions and figures. Temporal demand modeling does not transfer directly to radiographs; classification methods and grouped validation replace it.

## Literature review

[Kermany et al. (2018), Cell](https://doi.org/10.1016/j.cell.2018.02.010), *Identifying Medical Diagnoses and Treatable Diseases by Image-Based Deep Learning*, supplies the provenance and pediatric pneumonia research context. Its neural-network experiments motivate the classification problem; their performance cannot be equated with this experiment's different partitions and features.

[Dalal and Triggs (2005), CVPR](https://doi.org/10.1109/CVPR.2005.177), *Histograms of Oriented Gradients for Human Detection*, supports gradient orientation descriptors. Edges may encode anatomy and opacities, but also borders and acquisition artifacts. The descriptor here uses orientation interpolation and overlapping normalized blocks without spatial interpolation.

[Ojala, Pietikäinen and Mäenpää (2002), IEEE TPAMI](https://doi.org/10.1109/TPAMI.2002.1017623), *Multiresolution Gray-Scale and Rotation Invariant Texture Classification with Local Binary Patterns*, motivates local texture features. This implementation uses eight immediate square neighbors with rotation-invariant uniform coding; it does not reproduce circular interpolated radius sampling.

[Cortes and Vapnik (1995), Machine Learning](https://doi.org/10.1007/BF00994018), *Support-Vector Networks*, motivates margin-based classification. We evaluate a linear SVM because the engineered vector is relatively high-dimensional and sample size modest. It cannot learn all nonlinear structure, and its decision scores are uncalibrated.

[Zech et al. (2018), PLOS Medicine](https://doi.org/10.1371/journal.pmed.1002683), *Variable generalization performance of a deep learning model to detect pneumonia in chest radiographs*, shows the risk of radiological confounding and external performance loss. This motivates overlap controls and the explicit restriction of conclusions to the internal dataset.

The research question is whether inexpensive texture features add discriminative value beyond intensity summaries and a majority classifier. Feature and scaling ablations use identical development folds to test that question. Published headline accuracies are not used as equivalent baselines.

## Dataset quality and EDA

The canonical local tree contains 5,856 readable images, with no decoding failures. Train contains 1,341 NORMAL and 3,875 PNEUMONIA; validation has 8 of each; test has 234 NORMAL and 390 PNEUMONIA. This measured total is used instead of assuming a nominal publication count. A second nested dataset copy and macOS metadata are ignored.

The audit records original size, color mode, intensity mean/standard deviation, near-black/near-white pixel fractions, a decoded grayscale pixel SHA-256 and a filename patient proxy. There are 32 repeated pixel images, no label-conflicting hash groups, and no exact hash groups spanning the original splits. However, patient proxies connect development and test. The protocol excludes 457 development images linked to test, then removes 28 remaining duplicate copies. Test loses six duplicate copies. Final development contains 1,348 NORMAL and 3,405 PNEUMONIA; test contains 231 NORMAL and 387 PNEUMONIA.

These proxies parse `person...`, `IM-...` and `NORMAL2-IM-...` prefixes. They are not verified patient records. Conservatively excluding matching identifiers may remove unrelated images if identifiers collide; different identifiers may still belong to one patient. This protocol reduces known overlap but does not certify patient-disjoint evaluation. Exact hashes do not detect every recompressed or transformed near duplicate.

`eda_overview.png` shows supplied counts, original image dimension variability and mean intensity distributions. `eda_statistics.csv` contains measured descriptive statistics by class and split. `example_radiographs.png` uses seeded development examples only. Brightness/dimension patterns may reflect acquisition or processing differences; they are not evidence of causal disease markers. No demographic completeness or annotation correctness can be certified from the available files.

## Feature engineering and scaling

Every image is grayscale, resized without changing aspect ratio, and black-padded to 64×64. Fixed pixel normalization divides by 255. The resolution makes CPU experiments practical but sacrifices subtle detail and leaves padding as a possible shortcut signal. No augmentation or lung segmentation is used.

The 1,799-element vector includes 1,764 HOG values (nine unsigned bins, 8×8 cells, 2×2 overlapping blocks with L2-Hys), ten normalized uniform square-neighbor LBP bins, sixteen normalized intensity histogram bins and nine statistics (mean, standard deviation, five quantiles and mean absolute horizontal/vertical differences).

StandardScaler computes feature means and standard deviations only on a training fold: z=(x−μ_train)/σ_train. Scaled models are sklearn Pipelines, which refit scaling for every CV training partition and apply it unchanged to validation. Final models fit all development data and transform test using development statistics. The forest needs no scaling. Per-image histogram/block normalization does not learn cross-image statistics. Filenames, original dimensions, group IDs and split labels never enter the predictor vector.

See the [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html) and [grouped CV documentation](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.StratifiedGroupKFold.html) for the implementation rationale.

## Theory and experimental design

The majority classifier establishes the imbalance reference. Logistic regression minimizes class-weighted log loss with L2 regularization. The linear SVM uses the regularized squared hinge objective. Smaller C means stronger regularization for both. The forest averages randomized trees and tunes minimum leaf size to constrain complexity. All nontrivial models use balanced class weights computed during their individual fits.

The supplied 16-image validation set is too small for reliable model ranking, so it joins development. Three stratified group folds with seed 42 maintain disjoint connected groups while approximately preserving class prevalence. The exact fold assignments are saved. Logistic C candidates are 0.01, 0.1 and 1; SVM C candidates are 0.001, 0.01 and 0.1; the 200-tree forest tests minimum leaves of 2 and 5. Balanced accuracy selects the best setting and model. The test does not influence these choices.

Sensitivity=TP/(TP+FN), specificity=TN/(TN+FP), balanced accuracy=(sensitivity+specificity)/2, precision=TP/(TP+FP), and F1 is the harmonic mean of precision and recall. ROC-AUC ranks continuous scores across thresholds; average precision summarizes precision–recall and depends on prevalence. Accuracy alone is inadequate with imbalance.

The selected model's development OOF scores choose an F1-maximizing operating threshold. Hyperparameters are tuned before this OOF calculation, so development estimates remain selection-biased; this is not nested cross-validation. The held-out test assesses the fixed final choices. Both the predeclared default and OOF thresholds are reported. The F1 threshold is an academic choice rather than a clinically validated operating point. Decision scores are not calibrated probabilities.

## Measured results and interpretation

Development mean balanced accuracy is 0.9680 for logistic regression (C=0.01; fold SD 0.0035), 0.9636 for linear SVM (C=0.001), 0.9337 for the forest (minimum leaf 5), and 0.5000 for the majority classifier. Logistic regression is selected. Its mean training balanced accuracy is 0.9939; the difference from CV suggests some overfitting. The forest's training score is 0.9920 despite weaker validation performance.

On the 618 retained test images, the development-selected threshold −0.27093 gives accuracy 0.8204, balanced accuracy 0.7632, pneumonia precision 0.7816, recall 0.9897, specificity 0.5368, F1 0.8734, ROC-AUC 0.9628 and average precision 0.9771. The confusion counts are TN=124, FP=107, FN=4 and TP=383.

At its default zero decision threshold, the same model gives accuracy 0.8269, balanced accuracy 0.7736, recall 0.9845, specificity 0.5628 and F1 0.8769. The tuned threshold did not improve test F1, illustrating that threshold tuning on development need not generalize. We retain the declared threshold and report the difference rather than selecting a new threshold on test.

Group bootstrap 95% intervals at the selected threshold are approximately 0.7302–0.8005 for balanced accuracy, 0.9788–0.9976 for recall, 0.4684–0.6093 for specificity and 0.9468–0.9752 for ROC-AUC. These percentile intervals use 1,000 resamples of filename/hash test groups. They do not cover uncertainty from training/model selection, unrecognized patient identity or external deployment.

The substantial development-to-test balanced-accuracy decline is a central limitation. The high ROC-AUC and low specificity at the chosen threshold suggest a change in score/operating-point behavior; these metrics alone do not identify its cause. Possible explanations include acquisition differences, remaining confounding, limited patient proxies, and development selection bias. None is established causally by this experiment.

Fixed-C=0.1 ablations on identical development folds give balanced accuracy 0.9665 for scaled texture+intensity, 0.9569 without scaling, and 0.8864 for scaled intensity alone. Texture therefore adds approximately 8.01 percentage points over intensity alone in this control. Scaling adds approximately 0.97 points. These are development observations, not independent significance tests or separately tuned model comparisons.

The notebook shows the first eight test mistakes in manifest order. The examples support visual inspection of failure patterns but do not justify clinical interpretations or post-test model adjustment.

## Reproducibility and code quality

`src/pipeline.py` provides reusable stages and a CLI; the standalone notebook embeds that code so Colab needs no clone. The actual run saves manifests, exclusions, fold assignments, full CV grids, OOF scores, test predictions, confidence intervals, diagnostic figures, package versions and a joblib model plus threshold. Seed 42 is fixed. Data and generated binary caches/models are ignored by git. `verify_artifacts.py` validates group/hash separation, fold separation, feature validity and agreement between saved-model predictions and exported results.

Exact package versions are in `results/run_environment.json`; dependency ranges are in `requirements.txt`. Source and notebook generation are kept together. Browser display in Colab does not imply that a cloud runtime has performed training: saved results were generated locally from the user's downloaded data.

## Limitations and next phase

This is a pediatric benchmark with internal test data and unverified patient identity. Exact deduplication is incomplete near-duplicate detection. Hand-crafted, low-resolution features may exploit padding, text markers or acquisition differences. Labels are accepted from folders without independent clinical review. Probabilities are not calibrated, performance is not externally validated, and test specificity is weak. No clinical deployment claim is warranted.

If Phase 2 permits it, investigate external validation, reliable patient metadata, acquisition/marker confounding, lung-focused preprocessing, probability calibration using separate development data, nested CV, and higher-resolution or neural-network features. Any new choices should receive a new independent test; the current test has already been inspected.
