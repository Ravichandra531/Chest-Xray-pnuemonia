"""Build a standalone Colab notebook from the reviewed pipeline source."""
import json
from pathlib import Path

ROOT=Path(__file__).resolve().parent
cells=[]
def md(text): cells.append(dict(cell_type='markdown',metadata={},source=text))
def code(text): cells.append(dict(cell_type='code',metadata={},source=text,execution_count=None,outputs=[]))

md('''# Chest X-Ray Pneumonia Classification — Phase 1
## Classical Machine Learning: EDA, feature engineering, scaling and validation

**Question:** How well can reproducible texture and intensity features discriminate the dataset's NORMAL and PNEUMONIA labels?

This is a student ML benchmark, not a clinical diagnostic system. No neural networks are trained. Positive class: **PNEUMONIA = 1**.

The structure adapts the EDA → features → model evaluation organization of [Inventory-demand-forecasting](https://github.com/tejuuu-7774/Inventory-demand-forecasting). Forecasting methods and metrics are replaced with classification methods suited to radiographs.

**Run in Colab:** choose CPU; upload this notebook using File → Upload notebook. Put the extracted dataset at `/content/chest_xray`, or in Google Drive at `MyDrive/chest_xray`. Run all cells. A Mac Downloads folder is not directly accessible from Colab. Set `DATA_ROOT` below if your folder differs. This notebook embeds the implementation and does not require a GitHub clone.

**Evaluation coverage:** literature review; dataset quality and EDA; justified feature engineering; theoretical rigor; reproducible model comparison; saved artifacts and code quality.''')
md('''## 1. Literature review and research gap

1. [Kermany et al., Cell (2018)](https://doi.org/10.1016/j.cell.2018.02.010): introduces image-based diagnostic experiments including pediatric pneumonia. Establishes provenance and a deep-learning reference; it does not validate this classical ML implementation. The downloaded dataset is a pediatric, single-source benchmark, so generalization to adult or external-hospital images is unproven.
2. [Dalal & Triggs, CVPR (2005)](https://doi.org/10.1109/CVPR.2005.177): motivates gradient orientation histograms. Here HOG is a hand-crafted descriptor of edges, not a learned disease representation. Our explicitly documented implementation omits spatial vote interpolation.
3. [Ojala et al., IEEE TPAMI (2002)](https://doi.org/10.1109/TPAMI.2002.1017623): motivates local binary patterns for texture. We use the 8 immediate square neighbors, a documented variation from circular interpolated sampling.
4. [Cortes & Vapnik, Machine Learning (1995)](https://doi.org/10.1007/BF00994018): motivates margin-based SVM classification; linear SVM is appropriate for high-dimensional texture vectors.
5. [Zech et al., PLOS Medicine (2018)](https://doi.org/10.1371/journal.pmed.1002683): demonstrates hospital-specific confounding and degraded external performance in radiology models. Motivates leakage controls and cautious interpretation.

**Gap and hypothesis:** test whether low-cost, transparent texture descriptors improve over intensity alone and the majority baseline. Literature scores are not directly comparable unless patients, preprocessing, and partitions match. The feature/scaling ablation tests this hypothesis on development folds only.

[Dataset source](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data). Inspect the source's current license before redistribution; raw images are excluded from this repository.''')
code('''import sys, subprocess, importlib.util
from pathlib import Path
IN_COLAB = importlib.util.find_spec('google.colab') is not None
if IN_COLAB:
    subprocess.check_call([sys.executable, '-m', 'pip', 'install', '-q',
        'numpy>=2,<3', 'pandas>=2.2,<4', 'Pillow>=10,<13',
        'scikit-learn>=1.6,<1.10', 'matplotlib>=3.9,<4', 'joblib>=1.4,<2'])
print('Runtime:', 'Google Colab' if IN_COLAB else 'Local Python', sys.version)
''')
code('''# Edit only this cell to change input/output paths.
DATA_ROOT = Path('/content/chest_xray') if IN_COLAB else Path.home() / 'Downloads/chest_xray'
OUTPUT = Path('/content/pneumonia_phase1') if IN_COLAB else Path.cwd()
if IN_COLAB and not DATA_ROOT.exists():
    from google.colab import drive
    drive.mount('/content/drive')
    DATA_ROOT = Path('/content/drive/MyDrive/chest_xray')
if not DATA_ROOT.exists():
    raise FileNotFoundError(f'Upload/extract chest_xray or edit DATA_ROOT: {DATA_ROOT}')
for folder in ('Datasets/processed', 'results', 'report/figures', 'models'):
    (OUTPUT / folder).mkdir(parents=True, exist_ok=True)
print('Input:', DATA_ROOT, '\\nOutput:', OUTPUT)
''')
md('''## 2. Implementation and experimental protocol

The following cell contains the same implementation as `src/pipeline.py`, embedded for standalone Colab execution. It defines functions; later cells run the stages separately.

We scan exactly one canonical train/val/test tree, ignore macOS metadata, decode every image, fingerprint grayscale pixels, and form connected groups from filename proxies and duplicate hashes. Development groups touching test are excluded. Label-conflicting exact duplicates are excluded; other exact duplicates are retained once. The small supplied validation set joins development. Three stratified group folds select models and hyperparameters by balanced accuracy. Test labels are never used for tuning; structural overlap auditing necessarily inspects all file identities. All supervised development uses the retained development subset.

Grouping is conservative: IM identifiers are filename proxies, not independently verified patient IDs. Near duplicates and acquisition artifacts can remain. Population representativeness and annotation errors cannot be established from images alone.''')
source=(ROOT/'src/pipeline.py').read_text().split("if __name__=='__main__':")[0]
code(source)
md('''## 3. Dataset quality audit and EDA

Inspect measured class counts, decode failures, duplicate hashes, exclusions, image sizes and grayscale distributions. EDA is descriptive; it does not establish disease-specific biology. Test appearance summaries are shown for transparency, but preprocessing/model choices are fixed in advance. Report actual local counts rather than repeating a nominal count from a publication.''')
code('''root, manifest, audit_summary = audit(DATA_ROOT, OUTPUT)
print(json.dumps(audit_summary, indent=2))
eda(root, manifest, OUTPUT)
from IPython.display import display, Image as NotebookImage
display(manifest.groupby(['split', 'label']).size().rename('images').to_frame())
display(NotebookImage(filename=str(OUTPUT / 'report/figures/eda_overview.png')))
display(NotebookImage(filename=str(OUTPUT / 'report/figures/example_radiographs.png')))
''')
md(r'''## 4. Feature engineering and scaling

Grayscale images are aspect-preserving resized and black-padded to 64 × 64. This trades fine detail for CPU speed and avoids shape distortion. Padding and acquisition differences may themselves confound predictions. No lung segmentation or augmentation is applied.

Feature vector: 1,764 HOG + 10 uniform square-neighbor LBP + 16 intensity histogram + 9 summary statistics = **1,799 features**. No filenames, dimensions, split names or patient proxies enter predictors.

Pixel normalization is the fixed transform $x/255$; it learns nothing. Standardization learns $z_j=(x_j-\mu_{j,train})/\sigma_{j,train}$. A `Pipeline` fits the scaler separately inside each CV training fold, then uses that scaler on validation. Final scaling fits development only. Trees use raw feature vectors because monotonic scaling is unnecessary for their splits. HOG block normalization and histogram normalization are local per-image operations, distinct from dataset standardization.

The NumPy descriptor definitions are explicit so this project runs without a specialist image-feature dependency. They are inspired by the cited papers, not exact replicas of their complete implementations.''')
code('''retained, X = features(root, manifest, OUTPUT)
print('Feature matrix:', X.shape, 'finite:', np.isfinite(X).all())
print((OUTPUT / 'results/features.json').read_text())
''')
md(r'''## 5. Theory and model application

**Majority baseline:** predicts the majority development label. Accuracy can look high on imbalanced data, while balanced accuracy is 0.5 for a constant classifier with both classes present.

**Logistic regression:** $p(y=1|x)=\sigma(w^Tz+b)$; minimizes class-weighted log loss plus L2 regularization. Smaller `C` increases regularization. It is interpretable as a linear decision function, but correlated image features limit causal interpretation.

**Linear SVM:** minimizes a regularized squared hinge objective, $\frac12\|w\|^2+C\sum_i\alpha_{y_i}\max(0,1-t_i(w^Tz_i+b))^2$. Its decision score is not a calibrated probability. Margin control can limit overfitting in high-dimensional feature spaces.

**Random forest:** averages 200 randomized decision trees; nonlinear reference model with class weighting and leaf-size tuning. Scaling is unnecessary. Tree complexity and correlated artifacts may cause overfitting.

Use class weighting instead of duplicating images or applying SMOTE to image texture vectors. Hyperparameter candidates and folds are fixed with seed 42. Training-vs-CV scores reveal possible overfitting.

Balanced accuracy $=(sensitivity+specificity)/2$ is the selection criterion. Also report precision, recall, specificity, F1, ROC-AUC and average precision. Average precision summarizes a PR curve and is not trapezoidal PR-AUC. AUC uses continuous scores. A development OOF threshold maximizes F1; this is an academic operating point, not a clinical sensitivity requirement. The default operating point is also reported. Hyperparameters are selected before producing OOF predictions, so OOF tuning estimates are not unbiased nested-CV estimates. The final untouched test evaluates the fixed selection.

95% percentile bootstrap intervals resample test groups 1,000 times. They describe sampling variation within this dataset, conditional on the fitted model; they exclude training and deployment uncertainty.''')
code('''import time, platform, importlib.metadata
started = time.monotonic()
final_result = train(retained, X, OUTPUT)
save_json(dict(seed=SEED, python=platform.python_version(),
    packages={p:importlib.metadata.version(p) for p in ['numpy','pandas','Pillow','scikit-learn','matplotlib','joblib']},
    training_runtime_seconds=time.monotonic()-started), OUTPUT / 'results/run_environment.json')
print(json.dumps(final_result, indent=2))
''')
md('''## 6. Comparison, ablation and error analysis

Model ranking below uses development CV only. The test result is reported only for the selected model, preventing a second selection on test scores. The three ablations hold logistic-regression `C=0.1` fixed to isolate scaling/feature differences; they are not separately tuned competitors. Inspect false positives and false negatives without changing the final model after seeing them.''')
code('''display(pd.read_csv(OUTPUT / 'results/model_comparison_cv.csv'))
display(pd.read_csv(OUTPUT / 'results/feature_scaling_ablation.csv'))
print('Scaling check:', (OUTPUT / 'results/scaling_diagnostics.json').read_text())
display(NotebookImage(filename=str(OUTPUT / 'report/figures/final_evaluation.png')))
predictions = pd.read_csv(OUTPUT / 'results/test_predictions.csv')
errors = predictions[predictions.target != predictions.prediction]
print('Test errors:', len(errors))
fig, axes = plt.subplots(2, 4, figsize=(12, 7))
for ax in axes.ravel(): ax.axis('off')
for ax, row in zip(axes.ravel(), errors.head(8).itertuples()):
    with Image.open(root / row.path) as im: ax.imshow(im.convert('L'), cmap='gray')
    ax.set_title(f'True={row.target}; predicted={row.prediction}')
fig.suptitle('First eight test errors in manifest order; no clinical attribution')
fig.tight_layout()
error_path = OUTPUT / 'report/figures/error_examples.png'
fig.savefig(error_path, dpi=140); plt.close(fig)
display(NotebookImage(filename=str(error_path)))
''')
md('''## 7. Conclusions, limitations and viva preparation

Use the measured outputs above for your conclusions: compare the selected CV mean with the majority baseline; compare full features with intensity-only; assess whether scaling helped the fixed logistic model; contrast CV and held-out test balanced accuracy; discuss false negatives and specificity alongside accuracy.

**Limitations:** filename-derived grouping; internal single-source test; possible near duplicates and annotation errors; pediatric population; no external validation; low-resolution features; padding/markers/acquisition confounding; no calibrated probabilities; selection-biased development summaries; bootstrap intervals conditional on one training run. No claim of clinical readiness or performance parity with published neural networks is justified.

**Be ready to explain:** Why accuracy alone is inadequate; why normalization differs from standardization; why scaling before CV leaks information; why patient grouping matters; how class weights affect the objective; how `C` controls bias/variance; why SVM scores are not probabilities; why a test set cannot select thresholds; why external validation is necessary.

**Saved outputs:** manifest and exclusions, fixed fold assignments, full CV grids, feature/scaling ablations, OOF and test predictions, test metrics and confidence intervals, figures, package versions, fitted pipeline and threshold. Raw images and binary models are git-ignored; regenerate them with this notebook. Review `report/phase1_report.md` for the written evaluation narrative.''')
code('''# Optional Colab download: results, figures and trained model (no raw images).
import zipfile
archive = OUTPUT / 'phase1_results.zip'
with zipfile.ZipFile(archive, 'w', compression=zipfile.ZIP_DEFLATED) as z:
    for folder in ('results', 'report', 'models'):
        for p in (OUTPUT / folder).rglob('*'):
            if p.is_file(): z.write(p, p.relative_to(OUTPUT))
print('Saved:', archive)
# In Colab, uncomment to download:
# from google.colab import files
# files.download(str(archive))
''')
nb=dict(nbformat=4,nbformat_minor=5,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12'),colab=dict(name='Chest_Xray_Pneumonia_Phase1.ipynb',provenance=[])),cells=cells)
for i,c in enumerate(cells): c['id']=f'phase1-{i:02d}'
(ROOT/'Notebooks').mkdir(exist_ok=True)
(ROOT/'Notebooks/Chest_Xray_Pneumonia_Phase1.ipynb').write_text(json.dumps(nb,indent=1))
print('Notebook built:', len(cells), 'cells')
