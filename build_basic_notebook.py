"""Generate one self-contained, explainable Phase 1 notebook with explicit ML."""
import json
from pathlib import Path
import textwrap

ROOT = Path(__file__).resolve().parent
source = (ROOT / 'src/basic_ml.py').read_text()
cells = []
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s): cells.append(dict(cell_type='code',metadata={},source=textwrap.dedent(s).strip(),execution_count=None,outputs=[]))
def section(title, explanation, body=None):
    md(title+'\n\n'+explanation)
    if body is not None: code(body)

md('''# Pneumonia Classification from Chest X-Rays — Basic ML, Phase 1
## One notebook: literature → EDA → features → scaling → training → validation

**Aim:** distinguish the dataset's NORMAL and PNEUMONIA labels using a small, understandable classical ML pipeline.

**What is implemented from scratch:** patient-proxy grouping, fold assignment, feature extraction, scaling, majority baseline, Gaussian Naive Bayes, logistic regression, gradient descent, model selection, confusion counts, metrics, ROC curve and bootstrap intervals. **No scikit-learn, pandas, OpenCV, SciPy, TensorFlow, PyTorch, pretrained models or optimizer packages are used.**

**Supporting libraries:** Pillow only decodes/resizes JPEGs; NumPy supplies arrays and arithmetic; Matplotlib plots results. Python standard-library modules handle files, hashes, ZIPs and downloads. IPython's display utility is part of the notebook environment and only shows saved plots. These tools do not train the models for us. Strictly library-free Python cannot conveniently decode this JPEG dataset; the ML itself is explicit.

**How to run:** in Colab choose CPU and Run all. The data cell first looks for the image folder and, if absent, attempts the public Kaggle archive using Python's standard library. If Kaggle requires authentication, upload/extract your downloaded archive under `/content/data/chest_xray` and rerun. Local execution automatically uses `~/Downloads/chest_xray`. Results go to a separate `basic_phase1` folder; this experiment does not reuse the earlier sklearn scores.

**What to tell the instructor:** “We chose a simple supervised binary classifier, engineered interpretable image summaries, fitted scaling only on training folds, selected the model using grouped cross-validation, then evaluated a held-out test with multiple metrics.” This is a student benchmark, not a clinical diagnostic tool.''')

section('## 1. Literature review — what prior work teaches us','''[Kermany et al., Cell (2018)](https://doi.org/10.1016/j.cell.2018.02.010) motivates the pediatric pneumonia image classification task and provides dataset provenance. Its deep-learning experiments show the relevance of image-based classification, but their published results are not equivalent to our simpler features or our audited partitions.

[Ng & Jordan, NeurIPS (2001)](https://papers.nips.cc/paper_files/paper/2001/hash/7b7a53e239400a13bd6be6c91c4f6c4e-Abstract.html) compares discriminative logistic regression with generative Naive Bayes and discusses their different statistical behavior. It motivates our comparison; it does not guarantee that either wins on these images.

[Hastie, Tibshirani & Friedman, The Elements of Statistical Learning](https://hastie.su.domains/ElemStatLearn/main.html), especially the classification and model-assessment chapters, supplies the framework for regularization, validation and bias/variance. We apply those principles with explicit code.

[Zech et al., PLOS Medicine (2018)](https://doi.org/10.1371/journal.pmed.1002683) shows radiological models can exploit acquisition/hospital differences and lose performance externally. It motivates overlap auditing, cautious interpretation and the need for external validation.

**Our question:** do simple regional brightness/contrast features plus logistic regression outperform a majority baseline and Gaussian Naive Bayes? **Contribution for Phase 1:** explain every calculation, compare basic alternatives fairly, and report limitations rather than pursue a headline accuracy. No claims of matching published deep-learning results are made.

**Alternatives:** HOG/LBP could encode texture more richly, PCA could reduce features, SVM or a tree could model other boundaries, and CNNs could learn features automatically. We start with regional summaries and two basic models because their assumptions and training steps are easier to derive and defend. They trade representational power for simplicity.''')

section('## 2. Import only supporting libraries','''NumPy does vector arithmetic, Pillow reads pixels, and Matplotlib draws figures. The model code below never calls an ML estimator or library metric.''',r'''
import sys, csv, json, hashlib, re, zipfile, shutil, urllib.request, importlib.util, platform
from collections import Counter
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from IPython.display import display, Image as NotebookImage

SEED = 42
LABELS = {'NORMAL': 0, 'PNEUMONIA': 1}
IN_COLAB = importlib.util.find_spec('google.colab') is not None
print('Environment:', 'Google Colab' if IN_COLAB else 'Local Python')
print('NumPy is used for arithmetic; all ML operations are implemented below.')
''')
section('## 3. Find or download the dataset','''We use the actual downloaded images. Colab runs on a remote machine and cannot read a Mac Downloads folder. The optional public archive download avoids requiring an extra Kaggle client library. No credential is embedded. If downloading is blocked, the cell fails with a clear upload instruction; it never substitutes fabricated data.''',r'''
PROJECT_ROOT = Path('/content') if IN_COLAB else Path.cwd()
if PROJECT_ROOT.name == 'Notebooks': PROJECT_ROOT = PROJECT_ROOT.parent
OUTPUT = PROJECT_ROOT / 'basic_phase1'
OUTPUT.mkdir(parents=True, exist_ok=True)
FIGURES = OUTPUT / 'figures'
FIGURES.mkdir(exist_ok=True)
DATA_ROOT = Path('/content/data/chest_xray') if IN_COLAB else Path.home() / 'Downloads/chest_xray'

if IN_COLAB and not DATA_ROOT.exists():
    existing = [Path('/content/chest_xray'), Path('/content/drive/MyDrive/chest_xray')]
    DATA_ROOT = next((p for p in existing if p.exists()), DATA_ROOT)
if IN_COLAB and not DATA_ROOT.exists():
    archive = Path('/content/chest_xray_kaggle.zip')
    url = 'https://www.kaggle.com/api/v1/datasets/download/paultimothymooney/chest-xray-pneumonia'
    print('Downloading public Kaggle archive; this may take several minutes...', flush=True)
    if not archive.exists() or not zipfile.is_zipfile(archive):
        try:
            request = urllib.request.Request(url, headers={'User-Agent': 'Python phase1 educational notebook'})
            with urllib.request.urlopen(request, timeout=90) as response, archive.open('wb') as f:
                shutil.copyfileobj(response, f)
        except Exception as exc:
            raise RuntimeError('Download unavailable. Upload/extract your chest_xray archive under /content/data/chest_xray and rerun.') from exc
    if not zipfile.is_zipfile(archive):
        raise RuntimeError('Kaggle did not return a ZIP. Upload the downloaded dataset archive manually.')
    destination = Path('/content/data').resolve()
    destination.mkdir(exist_ok=True)
    with zipfile.ZipFile(archive) as z:
        # Extract only the shortest canonical tree, ignoring the nested duplicate.
        names = [n for n in z.namelist() if '/train/NORMAL/' in n and n.lower().endswith(('.jpeg','.jpg','.png')) and '__MACOSX' not in n]
        if not names: raise ValueError('Archive has no expected chest_xray/train/NORMAL images')
        prefix = min((n.split('/train/NORMAL/')[0] for n in names), key=len)
        for item in z.infolist():
            relative = item.filename[len(prefix)+1:] if item.filename.startswith(prefix+'/') else ''
            if relative.startswith(('train/','val/','test/')) and relative.lower().endswith(('.jpeg','.jpg','.png')) and not Path(relative).name.startswith('._'):
                target = (destination/'chest_xray'/relative).resolve()
                if not target.is_relative_to(destination): raise ValueError('Unsafe archive path')
                target.parent.mkdir(parents=True, exist_ok=True)
                with z.open(item) as src, target.open('wb') as dst: shutil.copyfileobj(src,dst)
    DATA_ROOT = destination/'chest_xray'
print('Dataset:', DATA_ROOT)
print('Output folder:', OUTPUT)
''')

section('## 4. Dataset quality — readable images, duplicates and group overlap','''**Why this step?** Training on duplicates or images of an apparent patient also present in test can inflate results. We decode one canonical tree, check pixel hashes and connect matching filename proxies or exact image hashes.

**Policy fixed before training:** keep test-linked groups out of development; remove label-conflicting exact duplicates; keep one remaining exact pixel copy. The tiny supplied validation folder joins development. Raw images are never changed.

**Alternative:** a random image split is simpler but may put repeated patient images into both partitions. A verified patient split is preferable, but verified identifiers are unavailable. Filename groups are only conservative proxies: collisions or unrecognized repeat patients can remain. Exact hashing does not identify every near duplicate.''',source[source.index('def write_csv'):source.index('def make_features')]+'''
root, records, failures = audit_images(DATA_ROOT)
write_csv(OUTPUT/'image_audit.csv', records)
if failures: write_csv(OUTPUT/'unreadable_images.csv', failures)
print('Readable images:', len(records), '| Decode failures:', len(failures))
print('Original split/class counts:', dict(Counter((r['split'],r['label']) for r in records)))
print('Repeated decoded-pixel images:', len(records)-len({r['pixel_hash'] for r in records}))
print('Excluded images:', dict(Counter(r['exclusion'] for r in records if r['exclusion'])))
retained = [r for r in records if r['partition']!='excluded']
print('Retained partition/class counts:', dict(Counter((r['partition'],r['label']) for r in retained)))
''')
section('## 5. EDA — sizes, missing metadata and class balance','''Count labels and examine dimensions before deciding how to represent an image. Clinical age/sex/site/verified patient fields are unavailable; they are not filled with guesses. Class imbalance motivates a majority baseline and balanced accuracy.''',r'''
def show_figure(filename):
    plt.tight_layout()
    plt.savefig(FIGURES/filename, dpi=130, bbox_inches='tight')
    plt.close()
    display(NotebookImage(filename=str(FIGURES/filename)))

for key in ('width','height','brightness','contrast'):
    values=np.array([r[key] for r in records])
    print(f'{key}: min={values.min():.2f}, median={np.median(values):.2f}, mean={values.mean():.2f}, max={values.max():.2f}')
print('Missing recorded metadata:', sum(v is None for r in records for v in r.values()))
print('Not supplied: clinical age, sex, acquisition site, verified patient identity.')
fig,axes=plt.subplots(1,2,figsize=(11,4))
splits=['train','val','test']; location=np.arange(3)
for j,label in enumerate(LABELS):
    values=[sum(r['split']==s and r['label']==label for r in records) for s in splits]
    axes[0].bar(location+(j-.5)*.35,values,width=.35,label=label)
    for ax,key in [(axes[1],'width')]:
        ax.hist([r[key] for r in records if r['label']==label],bins=30,alpha=.5,label=label)
axes[0].set_xticks(location,splits)
axes[0].set(title='Original class counts by split',xlabel='Supplied split',ylabel='Images (count)')
axes[1].set(title='Original image widths',xlabel='Width (pixels)',ylabel='Images (count)')
for ax in axes: ax.legend()
fig.suptitle('Source: canonical downloaded Kaggle chest_xray tree')
show_figure('01_class_balance_and_sizes.png')
''')
section('## 6. EDA — brightness, contrast and split differences','''Image acquisition can affect brightness and contrast. We describe these differences, without treating them as proven disease markers. Test summaries are an audit only; the feature set and experiment settings remain fixed. No test outcome chooses a model.''',r'''
fig,axes=plt.subplots(1,3,figsize=(15,4))
for label in LABELS:
    part=[r for r in records if r['label']==label]
    axes[0].hist([r['brightness'] for r in part],bins=30,density=True,alpha=.5,label=label)
    axes[1].hist([r['contrast'] for r in part],bins=30,density=True,alpha=.5,label=label)
for split in ['train','val','test']:
    values=[r['brightness'] for r in records if r['split']==split and r['label']=='NORMAL']
    axes[2].hist(values,bins=np.linspace(0,255,30),density=True,histtype='step',label=f'{split}, n={len(values)}')
axes[0].set(title='Brightness by class',xlabel='Mean intensity (0–255)',ylabel='Density (per intensity unit)')
axes[1].set(title='Contrast by class',xlabel='Pixel standard deviation',ylabel='Density (per intensity unit)')
axes[2].set(title='NORMAL brightness by split',xlabel='Mean intensity (0–255)',ylabel='Density (per intensity unit)')
for ax in axes: ax.legend()
show_figure('02_intensity_and_split_eda.png')
''')
section('## 7. EDA — inspect representative images','''Seeded development examples avoid cherry-picking unusually clear cases. Labels are inherited dataset annotations; these plots cannot independently verify diagnosis.''',r'''
rng=np.random.default_rng(SEED)
fig,axes=plt.subplots(2,3,figsize=(10,6))
for i,label in enumerate(LABELS):
    available=[r for r in retained if r['partition']=='development' and r['label']==label]
    chosen=rng.choice(len(available),size=3,replace=False)
    for ax,index in zip(axes[i],chosen):
        row=available[index]
        with Image.open(root/row['path']) as im: ax.imshow(im.convert('L'),cmap='gray')
        ax.set_title(f"{label}: {row['width']} × {row['height']} px"); ax.axis('off')
show_figure('03_example_images.png')
''')

section('## 8. Feature engineering — explainable regional summaries','''**Chosen:** grayscale, aspect-preserving resize/padding to 64×64, then divide pixel values by 255. Split the image into an 8×8 grid. For each block compute mean brightness and standard deviation: **64 means + 64 contrasts**. Add **16 histogram proportions + 5 global summaries = 149 features**.

**Why?** A radiograph contains spatial information; regional means retain coarse location while contrasts describe local variation. Histograms summarize the intensity distribution. These features have clear definitions and make basic ML training fast on CPU.

**Alternatives and tradeoffs:** raw 64×64 pixels give 4,096 correlated inputs; HOG/LBP add richer texture but more explanation and implementation; PCA reduces dimension but learns a less directly interpretable representation; CNNs learn features but exceed this basic Phase 1 scope. Regional summaries discard subtle details and can retain padding/marker shortcuts. No claim is made that they isolate pneumonia-specific pathology.

**Normalization vs scaling:** division by 255 is fixed per-image normalization. Standardization later learns training-set means and standard deviations. Filenames, original dimensions, group IDs and split labels are never predictors.''',source[source.index('def make_features'):source.index('def fit_scaler')]+'''
names=feature_names()
X=np.stack([make_features(root/r['path']) for r in retained])
y=np.array([r['target'] for r in retained],dtype=int)
groups=np.array([r['group'] for r in retained])
is_dev=np.array([r['partition']=='development' for r in retained])
assert X.shape[1]==len(names)==149 and np.isfinite(X).all()
print('Features:',X.shape,'| Label counts:',dict(Counter(y.tolist())))
print('Last five feature names:',names[-5:])
np.savez_compressed(OUTPUT/'basic_features.npz',X=X,y=y,groups=groups,is_development=is_dev)
''')
section('## 9. Training and validation design — protect the test','''We combine the tiny supplied validation folder with training into **development**. Three approximately stratified group folds compare models on the same rows. Each patient/hash proxy stays in one fold. The code shuffles deterministically, places larger groups first and balances each group's majority class; this is a simple explicit algorithm rather than an optimal stratification guarantee.

**Why three folds?** More robust than a single split while keeping the experiment small. Five-fold CV is a reasonable more expensive alternative. Nested CV would better estimate the selected model's development performance; we instead state that development scores are selection-biased and reserve the test for the fixed final choice.

**Fixed before training:** seed 42; logistic λ in {0.01, 0.1}; three folds; class-balanced training; selection by mean balanced accuracy; classification threshold 0.5. No test-tuned threshold is used.''',source[source.index('def fit_scaler'):source.index('def sigmoid')]+'''
Xd,yd,gd=X[is_dev],y[is_dev],groups[is_dev]
Xt,yt,gt=X[~is_dev],y[~is_dev],groups[~is_dev]
assert not set(gd)&set(gt)
assert not ({r['pixel_hash'] for r in retained if r['partition']=='development'} & {r['pixel_hash'] for r in retained if r['partition']=='test'})
folds,assignment=grouped_folds(yd,gd)
for number,(tr,va) in enumerate(folds):
    print(f'Fold {number}: train={len(tr)}, validation={len(va)}, validation positive fraction={yd[va].mean():.3f}')
write_csv(OUTPUT/'development_folds.csv',[dict(path=r['path'],group=r['group'],target=r['target'],fold=int(assignment[i])) for i,r in enumerate([r for r in retained if r['partition']=='development'])])
print('Development:',len(yd),'| Reserved test:',len(yt),'| Group overlap: 0')
''')
section('## 10. Feature scaling from scratch','''For each feature, compute **z = (x − training mean) / training standard deviation**. If a column is constant, use divisor 1 so it remains finite.

**Why standardize?** Features such as histogram proportions and regional contrast have different scales. Gradient descent and L2 regularization behave more consistently after standardization. **Alternative:** min–max scaling is understandable but sensitive to extreme values; no scaling leaves unequal numerical influence. Trees would not require standardization.

The demonstration fits fold 0 training rows only. Every model/CV fit later learns its own scaler; no globally standardized development matrix is reused across folds.''',r'''
tr,va=folds[0]
demo_scaler=fit_scaler(Xd[tr])
train_z=scale(Xd[tr],demo_scaler)
validation_z=scale(Xd[va],demo_scaler)
print('Largest absolute standardized training mean:',np.abs(train_z.mean(axis=0)).max())
print('Validation means need not be zero:',np.abs(validation_z.mean(axis=0)).mean())
for index in [0,64,128,144]:
    print(names[index],': original std=',round(Xd[tr,index].std(),4),', scaled std=',round(train_z[:,index].std(),4))
fig,axes=plt.subplots(1,2,figsize=(10,4))
columns=[0,64,128,144]
axes[0].boxplot(Xd[tr][:,columns],tick_labels=[str(c) for c in columns])
axes[1].boxplot(train_z[:,columns],tick_labels=[str(c) for c in columns])
axes[0].set(title='Before scaling: fold-0 training',xlabel='Feature index',ylabel='Original feature value')
axes[1].set(title='After scaling: fold-0 training',xlabel='Feature index',ylabel='Standardized value (unitless)')
show_figure('04_scaling_from_scratch.png')
''')

section('## 11. Theoretical rigor — logistic regression and gradient descent',r'''Logistic regression is a **classifier**, despite its name. It models a score $p=1/(1+e^{-(w^Tz+b)})$. We classify pneumonia when $p\ge0.5$.

Our exact objective is
$$J(w,b)=\frac1n\sum_i a_i[\log(1+e^{s_i})-y_i s_i]+\frac\lambda2\sum_j w_j^2,\quad s_i=w^Tz_i+b.$$
Class weights are $a_i=n/(2n_{y_i})$ and are computed from the current training fold. The bias is not penalized. Its gradients are
$$\nabla_wJ=Z^T[a\odot(p-y)]/n+\lambda w,\quad \partial J/\partial b=\operatorname{mean}(a\odot(p-y)).$$
Update $w\leftarrow w-\eta\nabla_wJ$ and similarly for $b$. Backtracking halves the learning rate when an update increases the objective. Stop at 1,000 iterations or improvement below $10^{-8}$. Record the objective so convergence is inspectable.

**Why this model?** A linear boundary, transparent loss, explicit gradients and a small number of settings make it a strong teaching baseline. L2 regularization discourages large coefficients and can reduce variance; larger λ means more shrinkage and potentially more bias. **Alternatives:** unregularized logistic regression may overfit; kNN avoids gradient optimization but requires distance/neighbor choices and more prediction-time work; SVM introduces margin loss; trees capture interactions but require split/depth choices.

**Why class weighting?** It gives each class equal total training influence without creating synthetic radiographs. **Alternatives:** oversampling duplicates training images, undersampling discards information, SMOTE creates synthetic feature vectors without assured radiological meaning. Class weighting changes the objective and probability interpretation: these scores are not established calibrated clinical probabilities.''',source[source.index('def sigmoid'):source.index('def train_naive_bayes')])
section('## 12. Model alternatives — majority and Gaussian Naive Bayes',r'''**Majority baseline:** always predict the most common development class. It shows why accuracy can look respectable without useful discrimination; balanced accuracy is 0.5 when both classes occur.

**Gaussian Naive Bayes:** estimate a mean and variance for each feature within each class, assume conditional feature independence, then sum Gaussian log-likelihoods and log priors. Use equal priors to represent equal class importance, and a fixed variance floor $10^{-3}$ in standardized squared units to prevent division by zero.

**Why include it?** It is a basic, fast model with closed-form training, contrasting generative assumptions with logistic regression's discriminative objective. **Limitation:** adjacent block features are correlated, so the independence assumption is approximate and scores may be overconfident. **Alternative:** empirical priors reflect development prevalence but emphasize its majority class. A full covariance Gaussian model represents correlations but estimates many more parameters.''',source[source.index('def train_naive_bayes'):source.index('def roc_points')])
section('## 13. Evaluation metrics — implement and understand them','''Positive class is PNEUMONIA. We calculate TP, TN, FP and FN explicitly.

- Accuracy: (TP+TN)/all cases.
- Recall/sensitivity: TP/(TP+FN); how many annotated pneumonia images are detected.
- Specificity: TN/(TN+FP); how many normal images are correctly recognized.
- Precision: TP/(TP+FP); fraction of positive predictions that match the annotation.
- F1: harmonic mean of precision and recall.
- Balanced accuracy: mean of recall and specificity; this is our model-selection metric.
- ROC-AUC: trapezoidal area under true-positive rate vs false-positive rate, grouping equal scores together.

**Why more than accuracy?** The labels are imbalanced and false positives/false negatives reveal different errors. Precision varies with prevalence. AUC measures ranking and does not establish a useful clinical operating point. **Alternative:** select by accuracy or F1, but either can obscure poor normal-class performance. A clinical sensitivity target would require domain validation; we do not invent one.''',source[source.index('def roc_points'):source.index('def grouped_bootstrap')])
section('## 14. Validate the mathematics before fitting images','''Check perfect, reversed and tied ROC-AUC; majority balanced accuracy; constant-column scaling; and loss reduction. A finite-difference check verifies the derived logistic gradient independently. These checks validate the implementation, not clinical usefulness.''',source[source.index('def verify_math'):]+"\nverify_math()")

section('## 15. Apply and compare models on identical development folds','''Compare the baseline, Gaussian NB and two regularization settings of logistic regression. The scaler and class weights are fitted on training rows **inside each fold**. Only held-out fold predictions contribute to the validation score. Save the training-vs-validation gap and logistic loss history.

The setting with the highest mean validation balanced accuracy wins. Ties keep the first declared candidate. These are selection scores rather than unbiased nested-CV performance estimates. The final test is untouched by ranking.''',r'''
candidates=[('majority',0.0),('gaussian_naive_bayes',0.0),('logistic_regression',0.01),('logistic_regression',0.1)]
fold_results=[]; comparisons=[]; logistic_histories={}
for name,lam in candidates:
    validation_scores=[]; training_scores=[]
    for fold,(tr,va) in enumerate(folds):
        scaler=fit_scaler(Xd[tr])
        x_train,x_validation=scale(Xd[tr],scaler),scale(Xd[va],scaler)
        model=train_model(name,x_train,yd[tr],regularization=lam)
        train_metric=metrics(yd[tr],predict_score(model,x_train))
        valid_metric=metrics(yd[va],predict_score(model,x_validation))
        training_scores.append(train_metric['balanced_accuracy'])
        validation_scores.append(valid_metric['balanced_accuracy'])
        row=dict(model=name,regularization=lam,fold=fold,training_balanced_accuracy=train_metric['balanced_accuracy'],**valid_metric)
        fold_results.append(row)
        if model['kind']=='logistic': logistic_histories[f'{name}, lambda={lam}, fold={fold}']=model['loss_history']
    comparison=dict(model=name,regularization=lam,cv_balanced_accuracy=float(np.mean(validation_scores)),cv_standard_deviation=float(np.std(validation_scores)),training_balanced_accuracy=float(np.mean(training_scores)))
    comparisons.append(comparison)
    print(comparison,flush=True)
winner=max(comparisons,key=lambda row:row['cv_balanced_accuracy'])
write_csv(OUTPUT/'model_comparison_cv.csv',comparisons)
write_csv(OUTPUT/'fold_metrics.csv',fold_results)
print('\nSelected from development CV only:',winner)
''')
section('## 16. Inspect optimization and generalization','''A decreasing training objective supports correct optimization but does not prove that a model generalizes. Compare training and CV balanced accuracy for overfitting. Fold-to-fold variation is descriptive; three overlapping-training folds are not independent trials for a significance claim.''',r'''
fig,axes=plt.subplots(1,2,figsize=(12,4))
for label,history in logistic_histories.items():
    axes[0].plot(np.arange(1,len(history)+1),history,label=label.split(', fold=')[0],alpha=.65)
axes[0].set(title='Logistic training objective by iteration',xlabel='Gradient-descent iteration',ylabel='Weighted log loss + L2 penalty')
handles,labels=axes[0].get_legend_handles_labels(); unique=dict(zip(labels,handles)); axes[0].legend(unique.values(),unique.keys(),fontsize=8)
locations=np.arange(len(comparisons))
axes[1].bar(locations-.17,[r['training_balanced_accuracy'] for r in comparisons],width=.34,label='Training')
axes[1].bar(locations+.17,[r['cv_balanced_accuracy'] for r in comparisons],width=.34,label='Validation CV')
axes[1].set_xticks(locations,[f"{r['model']}\nλ={r['regularization']}" for r in comparisons],rotation=20,ha='right')
axes[1].set(title='Training versus validation',xlabel='Declared model candidate',ylabel='Balanced accuracy (0–1)',ylim=(0,1)); axes[1].legend()
show_figure('05_training_and_cv.png')
''')
section('## 17. Fit the selected model and evaluate the held-out test','''Refit the scaler on all development rows, then train the selected model using only development labels. Evaluate test exactly at the **predeclared threshold 0.5**. The test score never changes the winner, features, λ or threshold.

These are results for an audited/deduplicated internal benchmark. They are not directly equivalent to results using the original unfiltered split, nor to external-hospital performance.''',r'''
final_scaler=fit_scaler(Xd)
final_model=train_model(winner['model'],scale(Xd,final_scaler),yd,regularization=winner['regularization'])
test_score=predict_score(final_model,scale(Xt,final_scaler))
test_prediction=(test_score>=.5).astype(int)
test_metrics=metrics(yt,test_score)
print('Selected model:',winner['model'],'lambda:',winner['regularization'])
for key,value in test_metrics.items(): print(f'{key}: {value:.4f}' if isinstance(value,float) else f'{key}: {value}')
test_records=[r for r in retained if r['partition']=='test']
write_csv(OUTPUT/'test_predictions.csv',[dict(path=r['path'],group=r['group'],target=r['target'],score=float(s),prediction=int(p)) for r,s,p in zip(test_records,test_score,test_prediction)])
''')
section('## 18. Inspect errors and ROC','''The confusion matrix reveals the operating-point tradeoff. Examine a fixed set of errors without retuning the model. The first eight mistakes in manifest order are not representative sampling of every failure mode, and we do not assign clinical explanations from image appearance alone.''',r'''
fig,axes=plt.subplots(1,2,figsize=(10,4))
matrix=np.array([[test_metrics['tn'],test_metrics['fp']],[test_metrics['fn'],test_metrics['tp']]])
axes[0].imshow(matrix,cmap='Blues')
for row in range(2):
    for col in range(2): axes[0].text(col,row,str(matrix[row,col]),ha='center',va='center',fontsize=14)
axes[0].set_xticks([0,1],['NORMAL','PNEUMONIA']); axes[0].set_yticks([0,1],['NORMAL','PNEUMONIA'])
axes[0].set(title='Test confusion matrix at threshold 0.5',xlabel='Predicted label',ylabel='Dataset label')
fpr,tpr=roc_points(yt,test_score)
axes[1].plot(fpr,tpr,label=f"Selected model; AUC={test_metrics['roc_auc']:.3f}")
axes[1].plot([0,1],[0,1],'--',label='Random ranking')
axes[1].set(title='Held-out test ROC curve',xlabel='False-positive rate (0–1)',ylabel='True-positive rate (0–1)'); axes[1].legend()
show_figure('06_test_evaluation.png')
error_indices=np.flatnonzero(test_prediction!=yt)
print('Test mistakes:',len(error_indices))
if len(error_indices):
    fig,axes=plt.subplots(2,4,figsize=(12,6))
    for ax in axes.ravel(): ax.axis('off')
    for ax,i in zip(axes.ravel(),error_indices[:8]):
        with Image.open(root/test_records[i]['path']) as im: ax.imshow(im.convert('L'),cmap='gray')
        ax.set_title(f'True={yt[i]}, predicted={test_prediction[i]}')
    show_figure('07_first_test_errors.png')
''')
section('## 19. Feature ablation — test whether spatial features add value','''Compare all 149 features with only the 21 global histogram/summary features. Hold logistic λ=0.01 fixed on the same development folds. This tests the value of spatial block summaries rather than changing models, folds and settings at once. It is a development-only control and does not revise the final model after test inspection.''',r'''
ablation=[]
for label,indices in [('all_149_features',np.arange(149)),('global_only_21_features',np.arange(128,149))]:
    scores=[]
    for tr,va in folds:
        a,b=Xd[tr][:,indices],Xd[va][:,indices]
        scaler=fit_scaler(a)
        model=train_logistic(scale(a,scaler),yd[tr],regularization=.01)
        scores.append(metrics(yd[va],predict_score(model,scale(b,scaler)))['balanced_accuracy'])
    row=dict(features=label,lambda_fixed=.01,mean_cv_balanced_accuracy=float(np.mean(scores)),fold_std=float(np.std(scores)))
    ablation.append(row); print(row)
write_csv(OUTPUT/'feature_ablation.csv',ablation)
''')
section('## 20. Quantify uncertainty with group resampling','''Resample held-out filename/hash groups 500 times and compute percentile 95% intervals. **Why groups?** Repeated images from one apparent patient may be correlated. **Limitations:** proxies may be wrong, and these intervals condition on one fitted model; they exclude training/model-selection uncertainty and external population shifts. More bootstrap replicates would reduce Monte Carlo error.''',source[source.index('def grouped_bootstrap'):source.index('def verify_math')]+'''
intervals=grouped_bootstrap(yt,test_score,gt)
for key,bounds in intervals.items(): print(key,': 95% interval',tuple(round(v,4) for v in bounds))
''')
section('## 21. Save and validate the final experiment','''Save numerical model parameters in NPZ/JSON rather than depending on a model-serialization library. Reload parameters and confirm scores agree. Record exact versions, settings, retained counts and metrics. No raw radiographs are included in the results archive.''',r'''
parameters={'scaler_mean':final_scaler['mean'],'scaler_std':final_scaler['std']}
if final_model['kind']=='logistic': parameters.update(w=final_model['w'],b=np.array(final_model['b']))
elif final_model['kind']=='naive_bayes': parameters.update(means=final_model['means'],variances=final_model['variances'],prior=final_model['prior'])
else: parameters.update(score=np.array(final_model['score']))
np.savez_compressed(OUTPUT/'model_parameters.npz',**parameters)
loaded=np.load(OUTPUT/'model_parameters.npz')
reloaded={'kind':final_model['kind']}
for key in loaded.files:
    if not key.startswith('scaler_'): reloaded[key]=loaded[key]
reloaded_scaler={'mean':loaded['scaler_mean'],'std':loaded['scaler_std']}
assert np.allclose(test_score,predict_score(reloaded,scale(Xt,reloaded_scaler)))
assert not set(gd)&set(gt)
result={'experiment':'basic_from_scratch_149_features','seed':SEED,'selected_model':winner,
        'test_threshold':.5,'development_images':len(yd),'test_images':len(yt),
        'test_metrics':test_metrics,'bootstrap_95_intervals':intervals,
        'libraries':{'numpy':np.__version__,'Pillow':importlib.metadata.version('Pillow') if hasattr(importlib,'metadata') else 'see runtime','matplotlib':matplotlib.__version__},
        'python':platform.python_version(),'grouping_caveat':'filename/hash groups, not verified patient IDs',
        'execution_environment':'Colab' if IN_COLAB else 'Local Python'}
(OUTPUT/'final_results.json').write_text(json.dumps(result,indent=2))
(OUTPUT/'feature_names.json').write_text(json.dumps(names,indent=2))
print('PASS: reloaded model scores agree; development/test groups are disjoint.')
print('Saved all outputs to:',OUTPUT)
''')
section('## 22. Conclusions based on this run','''Read the actual values below. A large CV-to-test gap or weak specificity must be discussed even if accuracy/AUC looks encouraging. The ablation provides evidence for the chosen representation within development; it is not an independent clinical validation.''',r'''
print(f"Selected {winner['model']} using mean development CV balanced accuracy {winner['cv_balanced_accuracy']:.3f}.")
print(f"Held-out test: accuracy={test_metrics['accuracy']:.3f}, balanced accuracy={test_metrics['balanced_accuracy']:.3f}, recall={test_metrics['recall']:.3f}, specificity={test_metrics['specificity']:.3f}, ROC-AUC={test_metrics['roc_auc']:.3f}.")
print(f"False negatives={test_metrics['fn']}; false positives={test_metrics['fp']}.")
print('Development feature ablation:',ablation)
print('These are internal benchmark results with filename-derived grouping, not evidence of clinical readiness.')
''')
section('## 23. Instructor explanation guide — choices, alternatives and limitations','''**Why classification?** The target is a categorical folder annotation. Regression would predict a continuous quantity and does not match the binary task.

**Why basic features instead of raw pixels?** The 149 summaries retain coarse spatial/intensity information, reduce computation and have explainable definitions. Raw pixels, richer texture descriptors or learned CNN features may preserve more detail, at greater complexity.

**Why logistic regression?** We can derive its weighted log-loss, regularizer, gradient and update. It provides a transparent linear baseline. Its linear boundary can miss interactions. Gaussian NB offers a simpler closed-form alternative with stronger distribution/independence assumptions; the data, rather than preference, selects between them.

**Why normalize and standardize?** Dividing by 255 fixes a pixel range; standardizing makes features comparable for optimization and L2 penalties. Only the latter estimates training-set statistics. Fitting it before splitting would leak validation/test information.

**Why group validation?** Images are not necessarily independent patients. Proxy grouping and exact deduplication reduce known leakage; verified patient metadata would be better. We cannot certify independence from filenames alone.

**Why not use the supplied validation set directly?** It has only 16 images. Three grouped development folds provide more observations for comparison. A single grouped holdout is simpler; five folds or nested CV offer different computational/estimation tradeoffs.

**Why λ tuning and class weighting?** λ controls coefficient shrinkage and bias/variance. Class weights balance training influence without synthesizing images. Both weights and scaling are learned per training fold. The tiny λ grid prioritizes an explainable experiment over exhaustive tuning.

**Why balanced accuracy and several test metrics?** Both classes matter despite imbalance. Recall alone may reward predicting almost everything as pneumonia. Specificity exposes the complementary failure. A fixed threshold prevents test-tuning; calibration or a clinically chosen threshold would require separate development and clinical evidence.

**How do we know the implementation works?** Finite-difference gradient checks, controlled AUC/scaling examples, decreasing optimization loss, group-isolation assertions, saved fold assignments and reloaded-model score agreement. These validate calculations, not real-world diagnostic safety.

**Evaluation history:** The supplied test images were also inspected in earlier project versions. They are excluded from all tuning calculations in this run, but are not a newly unseen research holdout; a fresh external test set is needed for stronger claims after iterative development.

**What remains limited?** Pediatric single-source data; unverified patient identity; incomplete near-duplicate detection; possible acquisition/marker/padding shortcuts; label noise; loss of fine image detail; correlated NB features; selection-biased development estimates; no external validation; no established probability calibration. Our features cannot establish that the model learned pathological rather than acquisition signals.

**What would we try next?** Reliable patient metadata, external validation, richer hand-crafted features and leakage-safe feature selection/PCA. Compare SVM/trees if the instructor allows additional methods. Use a fresh evaluation set for changes informed by the present test. Deep learning is outside this basic Phase 1 experiment.

**One-minute explanation:** “We audited the images, removed exact duplicates and test-linked proxy groups, summarized each image with 149 brightness/contrast features, and standardized features using training rows only. We implemented majority prediction, Gaussian Naive Bayes and regularized logistic regression ourselves. We compared them using the same grouped folds and balanced accuracy, refitted the winner on development, and reported accuracy, sensitivity, specificity, F1 and AUC on the reserved test. We checked gradients and saved-model predictions, and we reported the dataset and validation limitations.”''')
section('## 24. Download results (optional)','''The ZIP contains metrics, fold assignments, numerical parameters, figures and feature names. It excludes radiographs and the feature cache. In Colab, uncomment the last two lines to download it.''',r'''
archive=OUTPUT/'basic_phase1_results.zip'
with zipfile.ZipFile(archive,'w',compression=zipfile.ZIP_DEFLATED) as z:
    for path in OUTPUT.rglob('*'):
        if path.is_file() and path!=archive and path.name!='basic_features.npz': z.write(path,path.relative_to(OUTPUT))
print('Saved:',archive)
# from google.colab import files
# files.download(str(archive))
''')

for i,c in enumerate(cells): c['id']=f'basic-ml-{i:03d}'
nb=dict(nbformat=4,nbformat_minor=5,cells=cells,metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12'),colab=dict(name='Chest X-Rays Pneumonia.ipynb',provenance=[])))
path=ROOT/'Notebooks/Chest X-Rays Pneumonia.ipynb'
path.write_text(json.dumps(nb,indent=1))
print(path,len(cells),'cells')
