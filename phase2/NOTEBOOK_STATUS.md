# Deep-learning notebook

`Phase2_DL_Pneumonia.ipynb` contains data preparation, a custom CNN, ResNet18, EfficientNet-B0, a proposed hybrid classifier, ablation experiments, evaluation metrics, and Grad-CAM code.

The saved notebook shows three custom-CNN epochs and a two-stage ResNet18 training log. Its EfficientNet cell records a `NameError`. The hybrid, ablation, final comparison, and Grad-CAM sections have no saved outputs, so their results have not been established by this notebook.

The notebook evaluates on the test loader during training and selects checkpoints using those scores. It also plans model and threshold selection on the test set. Therefore its logged results are exploratory, not independent final-test estimates. For a fresh evaluation, use a separate validation split for development decisions and reserve a test set until the end.

The updated report is `reports/Chest_Xray_Deep_Learning_Report.pdf`. The notebook is included as supplied, including its saved outputs. Running it may require installing packages, downloading ImageNet weights, access to the chest X-ray dataset, and substantial GPU time.
