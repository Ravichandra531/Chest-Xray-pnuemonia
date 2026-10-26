"""Build three sequential teaching notebooks in the reference repository's style."""
import json
import textwrap
from pathlib import Path

ROOT=Path(__file__).resolve().parent
source=(ROOT/'src/pipeline.py').read_text()
allin=json.loads((ROOT/'Notebooks/Chest_Xray_Pneumonia_Phase1.ipynb').read_text())
cells=[]
def md(s): cells.append(dict(cell_type='markdown',metadata={},source=s))
def code(s): cells.append(dict(cell_type='code',metadata={},source=textwrap.dedent(s).strip(),execution_count=None,outputs=[]))
def step(n,title,why,body):
    md(f'## {n}. {title}\n\n{why}'); code(body)
def write(name):
    for i,c in enumerate(cells): c['id']=f'{name[:2]}-cell-{i:03d}'
    nb=dict(nbformat=4,nbformat_minor=5,cells=cells.copy(),metadata=dict(kernelspec=dict(display_name='Python 3',language='python',name='python3'),language_info=dict(name='python',version='3.12'),colab=dict(name=name,provenance=[])))
    (ROOT/'Notebooks'/name).write_text(json.dumps(nb,indent=1))
    print(name,len(cells),'cells'); cells.clear()

COMMON="""
import sys, json, hashlib, re, importlib.util
from pathlib import Path
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from PIL import Image, ImageOps
from IPython.display import display, Image as NotebookImage
SEED = 42
CLASSES = {'NORMAL': 0, 'PNEUMONIA': 1}
IN_COLAB = importlib.util.find_spec('google.colab') is not None
print('Environment:', 'Google Colab' if IN_COLAB else 'Local Python')
"""
PATHS="""
# Use the SAME PROJECT_ROOT in all three notebooks.
# Colab: put the extracted images under MyDrive/chest_xray before running.
if IN_COLAB:
    from google.colab import drive
    drive.mount('/content/drive')
    PROJECT_ROOT = Path('/content/drive/MyDrive/Chest-Xray-Pneumonia-ML')
    DATA_ROOT = Path('/content/drive/MyDrive/chest_xray')
else:
    PROJECT_ROOT = Path.cwd()
    if PROJECT_ROOT.name == 'Notebooks': PROJECT_ROOT = PROJECT_ROOT.parent
    DATA_ROOT = Path.home() / 'Downloads/chest_xray'
OUTPUT = PROJECT_ROOT
for folder in ('Datasets/processed','results','report/figures','models'):
    (PROJECT_ROOT / folder).mkdir(parents=True, exist_ok=True)
PROCESSED_DIR = PROJECT_ROOT / 'Datasets/processed'
FIGURE_DIR = PROJECT_ROOT / 'report/figures'
print('Project folder:', PROJECT_ROOT)
print('Raw data folder:', DATA_ROOT)
"""
PLOT_HELPER="""
def show_plot(name):
    path = FIGURE_DIR / name
    plt.gcf().text(.01, .005, 'Source: local Kaggle chest_xray download; supplied splits; exploratory summaries', fontsize=8)
    plt.tight_layout(rect=(0,.025,1,1))
    plt.savefig(path, dpi=140)
    plt.close()
    display(NotebookImage(filename=str(path)))
"""
AUDIT=source[source.index('def save_json'):source.index('def eda')]
FEATURES=source[source.index('def texture_features'):source.index('def scores')]

md('''# Chest X-Ray Pneumonia — Data Exploration

We explore the downloaded NORMAL/PNEUMONIA images before feature engineering and model training. Each numbered section explains one task and shows its output, following the organization of the supplied [EDA reference notebook](https://github.com/tejuuu-7774/Inventory-demand-forecasting/blob/main/Notebooks/01_data_exploration.ipynb).

Run notebooks **01 → 02 → 03**. In Colab, upload all three notebooks and place the extracted image folder in Google Drive at `MyDrive/chest_xray`. All notebooks use the same Drive project folder. Saved local outputs can be viewed immediately; rerunning in Colab needs the dataset. This notebook performs EDA only; training is in notebook 03.''')
step(1,'Import the libraries','Pandas builds an image metadata table, NumPy summarizes pixels, Pillow reads images, and Matplotlib creates plots.',COMMON+'\n'+PLOT_HELPER)
step(2,'Find the project and dataset folders','We select one train/val/test tree. The nested second copy and macOS resource files must not be counted twice.',PATHS+"\nprint('Dataset exists:', DATA_ROOT.exists())")
step(3,'Load and inspect every image','The helper decodes images and records dimensions, intensity, pixel hashes and filename proxies. Its grouping logic also prepares an auditable exclusion policy for later notebooks. No predictor is trained here.',AUDIT+"\nroot, images, audit_summary = audit(DATA_ROOT, OUTPUT)\nprint('Readable image table:', images.shape)\nprint('Corrupt images:', audit_summary['corrupt_images'])")
step(4,'Identify the available columns','Each row is one image. The label comes from its folder. The filename-derived group is an imperfect patient proxy and must never be used as an input feature.',"display(pd.DataFrame({'column': images.columns, 'dtype': images.dtypes.astype(str).values}))")
step(5,'Preview the image table','Inspect several records before aggregating. Paths are relative to the canonical dataset root.',"display(images[['path','split','label','width','height','mode','mean_intensity','patient_proxy']].head(8))")
step(6,'Check the sizes of the dataset splits','Count images in each class and supplied split. The validation folder may be too small for reliable model selection.',"counts = pd.crosstab(images.split, images.label).reindex(['train','val','test'])\ndisplay(counts.assign(total=counts.sum(axis=1)))\nprint('Total readable images:', len(images))")
step(7,'Check duplicate images','Equal decoded grayscale pixels, including shape, define exact duplicates. Different JPEG bytes can decode to the same image. Near duplicates are not fully detected.',"print('Repeated pixel images:', images.duplicated('pixel_hash').sum())\nprint('Label-conflicting hashes:', audit_summary['conflicting_pixel_hashes'])\ndisplay(images[images.duplicated('pixel_hash', keep=False)][['path','split','label','pixel_hash']].head(10))")
step(8,'Check image and patient identifiers','Paths should identify individual files. Filename proxies can connect images from the same apparent patient. Cross-split group links are a leakage concern, but proxy equality does not certify identity.',"print('Duplicate relative paths:', images.path.duplicated().sum())\nprint('Filename patient proxies:', images.patient_proxy.nunique())\nprint('Groups occurring in multiple supplied splits:', (images.groupby('group').split.nunique()>1).sum())\ndisplay(pd.crosstab(images.split, images.excluded_reason.replace('', 'retained')))")
step(9,'Check missing values and corrupt files','Image data has no spreadsheet-style blank cells; missingness here means unreadable images or missing recorded metadata. Clinical demographics and verified patient identifiers are unavailable.',"display(images.isna().sum().rename('missing_metadata_values').to_frame())\ncorrupt = pd.read_csv(OUTPUT / 'results/corrupt_images.csv')\nprint('Unreadable image files:', len(corrupt))\nprint('Age, sex, acquisition site and verified patient ID: not supplied as metadata.')")
step(10,'Check image dimensions and color modes','Variable image dimensions require a consistent input representation. Color modes are inspected before converting to grayscale.',"display(images[['width','height']].describe().round(2))\ndisplay(images['mode'].value_counts().rename('images').to_frame())\nprint('Nonpositive dimensions:', ((images.width<=0)|(images.height<=0)).sum())")
step(11,'Calculate class balance','Pneumonia prevalence affects accuracy and precision. Balanced accuracy gives equal weight to each class.',"class_counts = images.label.value_counts()\ndisplay(pd.DataFrame({'images': class_counts, 'percent': 100*class_counts/len(images)}).round(2))\nprint('Pneumonia/normal ratio:', round(class_counts['PNEUMONIA']/class_counts['NORMAL'],2))\nax=counts.plot.bar(color=['#526575','#ac6045'],figsize=(8,4))\nax.set(title='Class counts by supplied split',xlabel='Supplied split',ylabel='Images (count)')\nshow_plot('eda_class_balance.png')")
step(12,'Summarize image brightness and contrast','Mean intensity describes average brightness; pixel standard deviation is a simple contrast proxy. These are image statistics, not confirmed disease markers.',"stats_columns=['mean_intensity','std_intensity','dark_fraction','bright_fraction']\ndisplay(images.groupby('label')[stats_columns].agg(['mean','median','std','min','max']).round(3))\nimages.groupby(['split','label'])[stats_columns].describe().to_csv(OUTPUT/'results/eda_extended_statistics.csv')")
step(13,'Plot the image-size distributions','Inspect width, height and aspect ratio. Aspect-preserving resizing avoids stretching anatomy.',"fig,axes=plt.subplots(1,3,figsize=(13,4))\nfor label in CLASSES:\n    part=images[images.label==label]\n    for ax,column in zip(axes[:2],['width','height']): ax.hist(part[column],bins=35,alpha=.5,label=label)\n    axes[2].hist(part.width/part.height,bins=35,alpha=.5,label=label)\nfor ax,title,unit in zip(axes,['Image width','Image height','Image aspect ratio'],['Width (pixels)','Height (pixels)','Width / height (ratio)']):\n    ax.set(title=title,xlabel=unit,ylabel='Images (count)'); ax.legend()\nshow_plot('eda_image_sizes.png')")
step(14,'Plot intensity distributions by class','Comparing distributions can reveal large brightness/contrast differences and possible preprocessing confounding. Overlap also explains why a single cutoff may be insufficient.',"fig,axes=plt.subplots(1,2,figsize=(11,4))\nfor label in CLASSES:\n    part=images[images.label==label]\n    axes[0].hist(part.mean_intensity,bins=35,alpha=.5,label=label,density=True)\n    axes[1].hist(part.std_intensity,bins=35,alpha=.5,label=label,density=True)\nfor ax,title,xlabel in zip(axes,['Mean image brightness','Within-image contrast'],['Mean grayscale intensity (0–255)','Pixel standard deviation (intensity units)']):\n    ax.set(title=title,xlabel=xlabel,ylabel='Probability density (per intensity unit)'); ax.legend()\nshow_plot('eda_intensity_distributions.png')")
step(15,'View representative radiographs','Use seeded development examples rather than choosing unusually clear images. Dataset labels are accepted annotations; visual examples do not verify them.',"fig,axes=plt.subplots(2,4,figsize=(12,7))\nfor row,label in enumerate(CLASSES):\n    examples=images[(images.label==label)&(images.partition=='development')].sample(4,random_state=SEED)\n    for ax,item in zip(axes[row],examples.itertuples()):\n        with Image.open(root/item.path) as im: ax.imshow(im.convert('L'),cmap='gray')\n        ax.set_title(f'{label}: {item.width} × {item.height} px'); ax.axis('off')\nshow_plot('eda_representative_images.png')")
step(16,'Compare the supplied splits','The reference aggregates sales over dates. Here the relevant comparison is acquisition statistics across train/val/test, because there is no reliable clinical time axis. Summary differences are descriptive and do not tune the model.',"display(images.groupby(['split','label'])[['mean_intensity','std_intensity','width','height']].mean().round(2))\nfig,axes=plt.subplots(1,2,figsize=(11,4))\nfor ax,label in zip(axes,CLASSES):\n    for split in ['train','val','test']:\n        values=images[(images.label==label)&(images.split==split)].mean_intensity\n        ax.hist(values,bins=np.linspace(0,255,30),density=True,histtype='step',linewidth=1.5,label=f'{split} (n={len(values)})')\n    ax.set(title=f'{label}: brightness by supplied split',xlabel='Mean grayscale intensity (0–255)',ylabel='Probability density (per intensity unit)'); ax.legend()\nshow_plot('eda_split_comparison.png')")
step(17,'Inspect dark and bright pixel fractions','Near-black and near-white pixel fractions flag exposure/border variation. We use no automatic quality cutoff, because arbitrary removal could bias the benchmark.',"fig,axes=plt.subplots(1,2,figsize=(11,4))\nfor label in CLASSES:\n    part=images[images.label==label]\n    axes[0].hist(100*part.dark_fraction,bins=30,alpha=.5,label=label)\n    axes[1].hist(100*part.bright_fraction,bins=30,alpha=.5,label=label)\nfor ax,title in zip(axes,['Near-black pixels (intensity ≤5)','Near-white pixels (intensity ≥250)']):\n    ax.set(title=title,xlabel='Pixels in image (%)',ylabel='Images (count)'); ax.legend()\nshow_plot('eda_exposure_proxies.png')")
step(18,'Check correlations among image statistics','Correlations reveal redundant summaries and possible acquisition relationships. Only retained development data is used here. Correlation does not imply causation.',"numeric=['width','height','mean_intensity','std_intensity','dark_fraction','bright_fraction']\ncorr=images.loc[images.partition=='development',numeric].corr()\ndisplay(corr.round(2))\nfig,ax=plt.subplots(figsize=(8,6))\nheat=ax.imshow(corr,vmin=-1,vmax=1,cmap='coolwarm')\nax.set_xticks(range(len(numeric)),numeric,rotation=40,ha='right'); ax.set_yticks(range(len(numeric)),numeric)\nfor r in range(len(numeric)):\n    for c in range(len(numeric)): ax.text(c,r,f'{corr.iloc[r,c]:.2f}',ha='center',va='center',fontsize=8)\nax.set(title='Development image-statistic correlations',xlabel='Image statistic',ylabel='Image statistic')\nfig.colorbar(heat,ax=ax,label='Pearson correlation (unitless)')\nshow_plot('eda_correlations.png')")
step(19,'Save the cleaned analysis manifest','The audit keeps test-linked groups out of development and keeps one exact pixel copy. The small supplied validation set joins development. Raw images are never modified.',"retained=images[images.partition!='excluded']\nprint('Exclusions:', audit_summary['exclusions'])\ndisplay(pd.crosstab(retained.partition,retained.label))\nprint('Saved:', PROCESSED_DIR/'manifest.csv')\nsummary={'readable_images':len(images),'corrupt_images':len(corrupt),'duplicate_images':int(images.duplicated('pixel_hash').sum()),'retained_images':len(retained)}\n(OUTPUT/'results/eda_summary.json').write_text(json.dumps(summary,indent=2))")
md('''## 20. EDA findings and next steps

The measured outputs show class imbalance, heterogeneous image dimensions, repeated images and cross-split filename group links. Inspect the plotted distributions for acquisition differences rather than assuming they represent disease effects. The 16-image validation set is too small for robust ranking.

Notebook 02 will normalize pixels, extract texture/intensity features and demonstrate training-only scaling. Notebook 03 will compare classical models with grouped CV and a reserved test. Patient proxies remain uncertain; true patient independence and external validity cannot be certified. No dates, weekdays or clinical demographic values are invented to imitate the retail dataset.

Data source: [Kaggle](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia/data). Dataset provenance: [Kermany et al. (2018)](https://doi.org/10.1016/j.cell.2018.02.010).''')
write('01_data_exploration.ipynb')

md('''# Chest X-Ray Pneumonia — Feature Engineering and Scaling

We turn images into numerical features, inspect them, demonstrate feature scaling without leakage and save prepared data for notebook 03. This follows the small-step style of the [feature engineering reference](https://github.com/tejuuu-7774/Inventory-demand-forecasting/blob/main/Notebooks/02_feature_engineering.ipynb). Run notebook 01 first. All three Colab notebooks use the same Google Drive project folder. Pixel normalization is fixed; standardization learns training statistics.''')
step(1,'Import libraries','Pillow preprocesses images; NumPy computes explicit descriptors; scikit-learn handles grouping and scaling.',COMMON+"\nfrom sklearn.preprocessing import StandardScaler\nfrom sklearn.model_selection import StratifiedGroupKFold\nimport joblib\n"+PLOT_HELPER)
step(2,'Find the project folders','Use the same project folder as notebook 01 so this stage loads its audited manifest.',PATHS)
step(3,'Load the audited image records','Check that the EDA stage completed. Do not silently recompute or change its exclusions.',"manifest_path=PROCESSED_DIR/'manifest.csv'\nif not manifest_path.exists(): raise FileNotFoundError('Run notebook 01 first, using the same PROJECT_ROOT.')\nimages=pd.read_csv(manifest_path,keep_default_na=False)\naudit_summary=json.loads((OUTPUT/'results/dataset_audit.json').read_text())\nroot=DATA_ROOT\nif not (root/'train').exists(): root=root/'chest_xray'\nprint('Audited records:',images.shape)\ndisplay(pd.crosstab(images.partition,images.label))")
step(4,'Set the experiment settings','Keep seed, labels, image size, folds and feature definitions fixed before model evaluation.',"IMAGE_SIZE=(64,64)\nN_FOLDS=3\nconfig={'seed':SEED,'image_size':IMAGE_SIZE,'n_folds':N_FOLDS,'positive_class':'PNEUMONIA','selection_metric':'balanced_accuracy','feature_version':1}\nprint(json.dumps(config,indent=2))")
step(5,'Resize and normalize an example image','Preserve aspect ratio and pad to 64×64. Divide pixels by 255; this operation learns no dataset statistics.',"example=images[images.partition=='development'].iloc[0]\nwith Image.open(root/example.path) as im:\n    original=im.convert('L')\n    resized=ImageOps.pad(original,IMAGE_SIZE,method=Image.Resampling.BILINEAR,color=0)\n    normalized=np.asarray(resized,dtype=np.float32)/255\n    fig,axes=plt.subplots(1,2,figsize=(8,4))\n    axes[0].imshow(original,cmap='gray'); axes[0].set_title(f'Original: {original.size} pixels')\n    axes[1].imshow(normalized,cmap='gray',vmin=0,vmax=1); axes[1].set_title('Padded 64×64; pixels / 255')\n    for ax in axes: ax.axis('off')\nshow_plot('feature_preprocessing.png')\nprint('Normalized pixel range:', normalized.min(), normalized.max())")
step(6,'Define the image descriptors','HOG encodes local gradients; uniform square-neighbor LBP encodes texture; histograms and summaries encode intensity. The code explicitly states sampling and normalization. Padding/artifacts can remain confounders. No filename/metadata predictors are used.',source[source.index('def save_json'):source.index('def resolve_root')]+FEATURES)
step(7,'Extract features for the retained images','All features are computed per image. No dataset-fitted transform is applied at this stage. The 1,799 columns consist of 1,764 HOG, ten LBP, sixteen histogram and nine summary values.',"retained,X=features(root,images,OUTPUT)\ny=retained.target.to_numpy()\nprint('Features:',X.shape,'Labels:',y.shape)\nprint('All values finite:',np.isfinite(X).all())\ndisplay(pd.DataFrame(X[:5,-9:],columns=['mean','std','p10','p25','p50','p75','p90','vertical_difference','horizontal_difference']))")
step(8,'Separate development and test','This split is inherited from the audit. Test labels never select preprocessing, model settings or thresholds.',"dev=retained.partition.eq('development').to_numpy()\nX_dev,y_dev=X[dev],y[dev]\ngroups=retained.group.to_numpy()[dev]\nassert not set(groups)&set(retained.loc[~dev,'group'])\nprint('Development:',X_dev.shape,'Test:',X[~dev].shape)")
step(9,'Create grouped cross-validation folds','StratifiedGroupKFold approximately balances labels while keeping each filename/hash component within one fold. These are patient proxies, not certified identities.',"cv=list(StratifiedGroupKFold(N_FOLDS,shuffle=True,random_state=SEED).split(X_dev,y_dev,groups))\nfold_rows=[]\nfor fold,(tr,va) in enumerate(cv):\n    assert not set(groups[tr])&set(groups[va])\n    fold_rows.append({'fold':fold,'training_images':len(tr),'validation_images':len(va),'validation_pneumonia_percent':100*y_dev[va].mean()})\ndisplay(pd.DataFrame(fold_rows))")
step(10,'Fit a scaler on one training fold',r'For each feature, $z=(x-\mu_{train})/\sigma_{train}$. Fit only on fold 0 training rows. The model notebook refits a separate scaler inside every CV fit.',"train_idx,val_idx=cv[0]\nscaler=StandardScaler().fit(X_dev[train_idx])\nX_train_scaled=scaler.transform(X_dev[train_idx])\nX_val_scaled=scaler.transform(X_dev[val_idx])\nprint('Scaler training rows:',scaler.n_samples_seen_)\nprint('Largest absolute scaled training mean:',np.abs(X_train_scaled.mean(axis=0)).max())\nprint('Validation means need not be zero:',np.abs(X_val_scaled.mean(axis=0)).mean())")
step(11,'Compare features before and after scaling','Inspect statistics from the training fold only. StandardScaler changes feature scales; it does not make features normally distributed.',"columns=[1764,1774,1790,1798]\ncomparison=pd.DataFrame({'feature_index':columns,'before_mean':X_dev[train_idx][:,columns].mean(0),'before_std':X_dev[train_idx][:,columns].std(0),'after_mean':X_train_scaled[:,columns].mean(0),'after_std':X_train_scaled[:,columns].std(0)})\ndisplay(comparison.round(5))\nfig,axes=plt.subplots(1,2,figsize=(10,4))\naxes[0].boxplot(X_dev[train_idx][:,columns],tick_labels=[str(c) for c in columns]); axes[0].set_title('Training features before scaling')\naxes[1].boxplot(X_train_scaled[:,columns],tick_labels=[str(c) for c in columns]); axes[1].set_title('Training features after scaling')\nfor ax in axes: ax.set(xlabel='Feature index',ylabel='Feature value (original units / standardized units)')\nshow_plot('feature_scaling_comparison.png')")
step(12,'Save prepared data and settings','Notebook 03 receives raw per-image features so its Pipeline learns scaling independently within every training fold. The scaled fold arrays below are demonstration artifacts only.',"np.savez_compressed(PROCESSED_DIR/'fold0_scaling_demo.npz',X_train_scaled=X_train_scaled,X_validation_scaled=X_val_scaled,train_idx=train_idx,val_idx=val_idx)\njoblib.dump(scaler,OUTPUT/'models/fold0_scaler_demo.joblib')\n(PROCESSED_DIR/'phase1_config.json').write_text(json.dumps(config,indent=2))\nprint('Saved features.npz, retained_manifest.csv, phase1_config.json and fold-0 scaling demonstration.')")
md('''## 13. Feature engineering conclusions

We preserved aspect ratio, normalized pixels, computed 1,799 reproducible image features, verified group isolation and demonstrated training-only standardization. Random forests do not need scaling. Logistic regression and SVM use scaling inside their CV pipelines. Never train every model on the globally pre-scaled fold demonstration arrays.

References: [HOG](https://doi.org/10.1109/CVPR.2005.177), [LBP](https://doi.org/10.1109/TPAMI.2002.1017623), [scikit-learn leakage guidance](https://scikit-learn.org/stable/common_pitfalls.html). Our exact NumPy variants are documented in the descriptor function. Proceed to notebook 03.''')
write('02_feature_engineering.ipynb')

md('''# Chest X-Ray Pneumonia — Model Development

Compare a majority baseline, logistic regression, linear SVM and random forest on identical prepared features and grouped folds. This adapts the [model-development reference](https://github.com/tejuuu-7774/Inventory-demand-forecasting/blob/main/Notebooks/03_model_development.ipynb) to binary image classification. Run notebooks 01 and 02 first. No deep-learning training is included.''')
imports=source[source.index('import argparse'):source.index('def save_json')]
step(1,'Import model libraries','Use scikit-learn pipelines, metrics and group CV; save real predictions and fitted models.',COMMON+'\n'+imports)
step(2,'Find the prepared-data and output folders','All stages use one project folder. Colab stages share outputs through Drive.',PATHS)
step(3,'Load prepared features and settings','Load exactly the row order saved by notebook 02. Retained metadata controls splitting, not prediction.',"config=json.loads((PROCESSED_DIR/'phase1_config.json').read_text())\nselected=pd.read_csv(PROCESSED_DIR/'retained_manifest.csv',keep_default_na=False)\ncache=np.load(PROCESSED_DIR/'features.npz')\nx=cache['X']\nassert np.array_equal(cache['y'],selected.target)\nassert config['seed']==SEED and config['n_folds']==3\nout=OUTPUT\nprint('Feature matrix:',x.shape)\nprint(json.dumps(config,indent=2))")
step(4,'Define the evaluation metrics','Balanced accuracy selects models. Also measure recall, specificity, precision, F1, ROC-AUC and average precision. Bootstrap groups, not independent images, for conditional uncertainty intervals.',source[source.index('def save_json'):source.index('def resolve_root')]+source[source.index('def scores'):source.index('def train')])
body=textwrap.dedent(source[source.index('def train'):source.index('def run')].split('\n',1)[1])
def segment(start,end): return body[body.index(start):body.index(end)]
step(5,'Set up development data and grouped folds','Keep the final test aside. Verify group disjointness and save exact validation-fold assignments.',body[:body.index('candidates =')])
step(6,'Define the baseline and three ML models','Logistic regression uses class-weighted L2 log loss; linear SVM uses a regularized squared hinge loss; the forest averages 200 trees. Smaller C means stronger regularization. Scaling is learned within each Pipeline fit. Class weights address imbalance.',segment('candidates =','rows, fitted ='))
step(7,'Train and tune using development CV','Search only the fixed candidate settings. Rank by mean balanced accuracy. Full search results include fold variation, training scores, ROC-AUC and recall.',segment('rows, fitted =','comparison ='))
step(8,'Compare models and select the winner','The majority baseline shows the effect of imbalance. Training–validation gaps flag overfitting. The test never ranks models.',segment('comparison =','    # Final selection'.strip())+"\ndisplay(comparison)")
step(9,'Choose a decision threshold from development predictions','OOF predictions select an F1 operating point. Hyperparameters were already selected on these folds, so development estimates are selection-biased and are not nested CV. This is not a clinically validated threshold. SVM and logistic decision scores are not calibrated probabilities.',segment('winner =','raw = getattr')+"\nprint('Selected model:',winner,'OOF F1 threshold:',threshold)")
step(10,'Evaluate the fixed final model on test','All choices are now fixed. Report both the default and declared OOF threshold; do not replace the latter after inspecting test performance. Save group-bootstrap intervals, predictions and the fitted pipeline.',segment('raw = getattr','fig, axes = plt.subplots(1,3')+"\nprint(json.dumps(result,indent=2))")
step(11,'Plot the confusion matrix, ROC and precision–recall','False negatives and false positives have different implications. A high AUC does not guarantee good specificity at a chosen threshold.',segment('fig, axes = plt.subplots(1,3','    # Feature ablation'.strip())+"\ndisplay(NotebookImage(filename=str(FIGURE_DIR/'final_evaluation.png')))")
step(12,'Compare feature and scaling controls','Keep logistic C=0.1 fixed and compare the same development folds. These controls isolate features/scaling; they are not independently tuned models and never alter the selected model.',segment('ablations=[]','    # Illustrate only'.strip())+"\ndisplay(pd.DataFrame(ablations))")
step(13,'Verify the scaling diagnostics','The first-fold training means should be approximately zero after scaling. Validation means can differ. Final model scalers use development only.',body[body.index('tr, va = cv[0]'):body.index('return result')]+"\nprint((out/'results/scaling_diagnostics.json').read_text())")
step(14,'Inspect the saved test mistakes','Review a fixed set of test errors descriptively. Do not use these examples to retune the final model.',"predictions=pd.read_csv(out/'results/test_predictions.csv')\nerrors=predictions[predictions.target!=predictions.prediction]\nprint('Test errors:',len(errors))\ndisplay(errors.head(8))\nerror_image=FIGURE_DIR/'error_examples.png'\nif error_image.exists(): display(NotebookImage(filename=str(error_image)))")
step(15,'Save the execution environment','Record package versions and configuration to make the experiment reviewable. Compatible dependency ranges are provided in requirements.txt.',"save_json({'seed':SEED,'python':platform.python_version(),'packages':{p:importlib.metadata.version(p) for p in ['numpy','pandas','Pillow','scikit-learn','matplotlib','joblib']}},out/'results/run_environment.json')\nprint('Saved model:',out/'models/best_model.joblib')\nprint('Saved evaluation:',out/'results/final_evaluation.json')")
md('''## 16. Interpret the results and limitations

Compare CV balanced accuracy with held-out test balanced accuracy. Discuss the measured recall/specificity tradeoff and whether the OOF F1 threshold actually improved test F1. The fixed-C ablation measures the added value of texture and scaling on development only.

The existing run selected logistic regression. It achieved high test ROC-AUC but substantially lower specificity and balanced accuracy than the development estimates. This is a limitation to explain, not hide. Filename proxies cannot certify patient independence. The data is pediatric and single-source; acquisition artifacts and near duplicates can remain. Confidence intervals are conditional on this trained model and cannot establish external generalization. This student benchmark is not clinically validated.

For the literature review and full theory, see the companion all-in-one Phase 1 notebook and `report/phase1_report.md`. [Kermany et al.](https://doi.org/10.1016/j.cell.2018.02.010), [Cortes & Vapnik](https://doi.org/10.1007/BF00994018), and [Zech et al.](https://doi.org/10.1371/journal.pmed.1002683) motivate the task, margin-based model and external-validation caveat.''')
write('03_model_development.ipynb')
