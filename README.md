# Chest X-Ray Pneumonia Classification — Phase 1

## Latest deliverable: one notebook, basic ML from scratch

Use **[Chest X-Rays Pneumonia.ipynb](Notebooks/Chest%20X-Rays%20Pneumonia.ipynb)** for the latest requested scope. It combines literature review, dataset quality/EDA, feature engineering, scaling, theory, basic-model comparison, validation and an instructor explanation guide in one executed notebook.

This version uses **NumPy for arithmetic, Pillow for JPEG decoding and Matplotlib for plotting only**. Scaling, group folds, Gaussian Naive Bayes, regularized logistic regression, gradient descent, metrics, AUC and bootstrap intervals are explicitly implemented. It imports no sklearn, pandas, SciPy or deep-learning library. Every choice includes a reason, an alternative and a limitation. In Colab it attempts a public dataset download when images are absent, avoiding the earlier missing-Drive-folder issue; if Kaggle denies the download, upload/extract the archive at the location indicated by the data cell.

All 22 code cells executed locally without errors. The selected from-scratch logistic model achieved 83.50% test accuracy, 79.06% balanced accuracy, 96.64% recall, 61.47% specificity and 0.9240 ROC-AUC on 618 retained test images. These are separate results from the earlier richer sklearn benchmark. Reproduce with the notebook and `requirements-basic.txt`; verify with `python verify_basic.py`. Outputs are under `basic_phase1/`. The core implementation is in `src/basic_ml.py`; the standalone notebook embeds it and needs no project imports.

The sections below document the earlier three-notebook/richer-feature version, retained for reference.

Classical machine learning benchmark on the locally downloaded [Kaggle chest X-ray dataset](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data). The project covers literature, EDA, image quality checks, hand-crafted features, leakage-aware scaling, grouped cross-validation, hyperparameter selection, held-out testing and error analysis.

The requested three-notebook sequence is:

**[Open the uploaded Google Colab copies](COLAB_LINKS.md)**. These contain actual local execution outputs; no cloud training run is claimed.

1. **[01_data_exploration.ipynb](Notebooks/01_data_exploration.ipynb)** — 20 numbered EDA sections with data checks, distributions, sample radiographs and saved findings.
2. **[02_feature_engineering.ipynb](Notebooks/02_feature_engineering.ipynb)** — image normalization, texture/intensity features, group folds, visible scaling demonstration and saved feature files.
3. **[03_model_development.ipynb](Notebooks/03_model_development.ipynb)** — model definitions, separate training/tuning steps, selection, test evaluation, plots and ablation.

These follow the short explanation → code → output style of the three supplied reference notebooks. Run them in order. In [Google Colab](https://colab.research.google.com/), choose **Upload notebook** and upload each file. Put extracted images in `MyDrive/chest_xray`; stages share outputs in `MyDrive/Chest-Xray-Pneumonia-ML`. Each notebook mounts Drive and uses the same project path. CPU is sufficient. Saved local outputs can be viewed without rerunning training.

The **[all-in-one notebook](Notebooks/Chest_Xray_Pneumonia_Phase1.ipynb)** additionally includes the full literature review, theory and complete workflow in one file. It also accepts `/content/chest_xray` for a temporary Colab runtime. Colab cannot directly access a Mac Downloads folder.

Inspired by the project organization of [Inventory-demand-forecasting](https://github.com/tejuuu-7774/Inventory-demand-forecasting); this project replaces forecasting with binary image classification. No neural-network training is included in Phase 1.

## Measured findings

- 5,856 images decoded; no corrupt files in the canonical tree. The download also contains a second nested copy, which is ignored.
- Original train: 1,341 normal / 3,875 pneumonia; validation: 8 / 8; test: 234 / 390.
- 32 repeated decoded-pixel images. No exact hashes crossed supplied splits, but filename-derived groups did.
- Removed 457 development images with group links to test and 28 additional duplicate images (including six test duplicates). Retained 4,753 development and 618 test images.
- Compared a majority baseline, logistic regression, linear SVM and random forest using fixed 3-fold stratified group CV.
- Logistic regression with `C=0.01` selected by development balanced accuracy: 96.80% mean CV balanced accuracy.
- Final test at the development OOF F1 threshold: **82.04% accuracy, 76.32% balanced accuracy, 98.97% pneumonia recall, 53.68% specificity, 0.9628 ROC-AUC**. There were 4 false negatives and 107 false positives.
- At the default threshold: 82.69% accuracy and 77.36% balanced accuracy. Both operating points were predeclared; the default result does not replace the selected threshold after test inspection.
- Fixed-`C` ablation CV balanced accuracy: scaled texture+intensity 96.65%; unscaled 95.69%; scaled intensity alone 88.64%.

The sizable CV-to-test gap and weak test specificity are central findings. These results support a student benchmark, not clinical deployment. Filename grouping is a proxy and cannot certify true patient independence. See [the written report](report/phase1_report.md).

## Repository organization

The main notebook is [`Chest X-Rays Pneumonia.ipynb`](Notebooks/Chest%20X-Rays%20Pneumonia.ipynb), also available in [Google Colab](https://colab.research.google.com/drive/1YYoOyMphKF56mTT9NHhXdePRvVKolE9e). Its results and parameters are in `basic_phase1/`; install `requirements-basic.txt` and run `python verify_basic.py` to verify a local run.


```text
Notebooks/Chest_Xray_Pneumonia_Phase1.ipynb  Standalone, executed notebook
Notebooks/01_data_exploration.ipynb         Detailed EDA in reference style
Notebooks/02_feature_engineering.ipynb      Feature extraction and scaling
Notebooks/03_model_development.ipynb        Step-by-step model development
src/pipeline.py                            Reusable audit, EDA, features and training
build_notebook.py                          Regenerate notebook from source
execute_notebook.py                        Local Python cell executor with real outputs
build_three_notebooks.py                   Regenerate the three teaching notebooks
verify_artifacts.py                        Leakage and saved-model consistency checks
Datasets/raw/                              Placeholder; never copy images into git
Datasets/processed/                        Audit manifests and generated feature cache
results/                                  CV grids, predictions, metrics, configuration
report/phase1_report.md                    Literature, theory, findings and limitations
report/figures/                            Actual EDA and evaluation plots
models/                                   Generated fitted pipeline (git-ignored)
requirements.txt                          Compatible dependency ranges
```

## Reproduce locally

Python 3.12 was used; exact measured package versions are in `results/run_environment.json`.

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m src.pipeline --data-root "$HOME/Downloads/chest_xray" --output .
python verify_artifacts.py
```

To regenerate and execute the full notebook locally, run `python build_notebook.py` then `python execute_notebook.py` from this directory. To generate the three-file sequence, run `python build_three_notebooks.py`, then `python execute_notebook.py Notebooks/01_data_exploration.ipynb` and repeat for 02 and 03, in order. Rebuilding clears old notebook outputs. The local executor needs IPython (included through ipykernel); it executes cells directly without starting a Jupyter server. Colab uses its standard Python kernel. Results may vary slightly across package versions and platforms.

## Protocol and reproducibility

Exactly one data tree is scanned. Grayscale pixel hashes and filename proxies form connected groups. Development images in groups touching test are excluded. Remaining exact duplicates are kept once. The tiny validation folder is pooled into development; all model selection uses stratified group CV. No predictor uses filename, split, original dimensions or labels embedded in names.

64×64 padded grayscale images yield 1,799 HOG/LBP/intensity features. NumPy implementations are explicit: orientation-interpolated HOG without spatial vote interpolation, and uniform LBP over immediate square neighbors. These are documented variants, not exact reproductions of every original descriptor detail. Pixel division by 255 is fixed normalization; StandardScaler learns development statistics separately inside each CV fold. Forest inputs do not need scaling.

Class weighting handles imbalance. Seed 42, exact folds, full hyperparameter grids, development OOF scores and final predictions are saved. Selection uses balanced accuracy; the F1 threshold is selected on development OOF scores. Because hyperparameters are tuned first, development estimates are selection-biased and are not nested-CV estimates. The test is used only after choices are fixed. Bootstrap intervals resample filename/hash groups and exclude training uncertainty.

## Phase 1 evaluation checklist

- **Literature review:** five primary references, relevance and applicability limits in the notebook/report.
- **Dataset quality & EDA:** image decoding, counts, dimensions, brightness, duplicate and overlap auditing, visual examples.
- **Feature engineering:** explicit HOG/LBP/intensity definitions, aspect preservation, no metadata predictors, feature ablation.
- **Theoretical rigor:** objectives, regularization, class weights, scaling equation, metric and threshold rationale.
- **Model application:** baseline and three classical models, grouped CV, saved searches, final metrics/intervals/errors.
- **GitHub & code quality:** modular source, standalone notebook, setup instructions, ignored data/model binaries and verification script. The repository includes source, executed notebooks, saved results and verification scripts.

## Sources and data handling

Primary sources and clickable citations are in the notebook and report. Raw images are not redistributed; check the dataset source's current terms before uploading data elsewhere. Generated joblib artifacts should only be loaded from trusted runs.
