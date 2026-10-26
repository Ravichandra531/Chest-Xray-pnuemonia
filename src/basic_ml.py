"""Basic Phase 1 ML implemented explicitly; no ML/statistics framework.

Supporting libraries: NumPy arrays, Pillow JPEG decoding, Matplotlib figures.
Everything else uses Python's standard library. The standalone notebook embeds
this implementation; this module supports the saved-artifact verification script.
"""
import csv
import hashlib
import json
import re
from pathlib import Path

import numpy as np
from PIL import Image, ImageOps

SEED = 42
LABELS = {"NORMAL": 0, "PNEUMONIA": 1}


def write_csv(path, rows):
    if not rows:
        return
    with Path(path).open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def patient_proxy(name):
    """A conservative filename grouping, NOT verified clinical identity."""
    stem = Path(name).stem
    match = re.match(r"(person\d+)_", stem)
    if match:
        return match.group(1)
    match = re.match(r"((?:NORMAL2-)?IM-\d+)", stem)
    return match.group(1) if match else "unknown:" + stem


def find_root(path):
    path = Path(path).expanduser().resolve()
    for candidate in (path, path / "chest_xray", path / "chest_xray/chest_xray"):
        if all((candidate / split / label).is_dir()
               for split in ("train", "val", "test") for label in LABELS):
            return candidate
    raise FileNotFoundError(f"Expected train/val/test, each with NORMAL/PNEUMONIA: {path}")


def audit_images(data_root):
    """Decode exactly one tree, identify duplicate pixels and link groups."""
    root = find_root(data_root)
    records, failures = [], []
    for split in ("train", "val", "test"):
        for label, target in LABELS.items():
            for path in sorted((root / split / label).iterdir()):
                if path.suffix.lower() not in (".jpeg", ".jpg", ".png") or path.name.startswith("._"):
                    continue
                try:
                    with Image.open(path) as im:
                        im.load()
                        pixels = np.asarray(im.convert("L"))
                        digest = hashlib.sha256(str(pixels.shape).encode() + pixels.tobytes()).hexdigest()
                        records.append(dict(path=str(path.relative_to(root)), split=split,
                            label=label, target=target, width=im.width, height=im.height,
                            mode=im.mode, brightness=float(pixels.mean()),
                            contrast=float(pixels.std()), dark_fraction=float((pixels <= 5).mean()),
                            bright_fraction=float((pixels >= 250).mean()),
                            pixel_hash=digest, patient_proxy=patient_proxy(path.name)))
                except (OSError, ValueError) as exc:
                    failures.append(dict(path=str(path.relative_to(root)), error=str(exc)))
    if not records:
        raise ValueError("No readable images")
    parent = list(range(len(records)))
    def find(i):
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i
    # Connected groups prevent a chain of duplicate/patient links crossing folds.
    for field in ("patient_proxy", "pixel_hash"):
        seen = {}
        for i, row in enumerate(records):
            key = row[field]
            if key in seen:
                parent[find(i)] = find(seen[key])
            else:
                seen[key] = i
    hash_labels = {}
    for i, row in enumerate(records):
        row["group"] = f"group_{find(i)}"
        hash_labels.setdefault(row["pixel_hash"], set()).add(row["target"])
    conflicts = {h for h, labels in hash_labels.items() if len(labels) > 1}
    test_groups = {r["group"] for r in records if r["split"] == "test"}
    seen_hashes = set()
    for row in records:
        reason = ""
        if row["split"] != "test" and row["group"] in test_groups:
            reason = "group_overlap_with_test"
        if row["pixel_hash"] in conflicts:
            reason = "conflicting_duplicate_label"
        if not reason:
            if row["pixel_hash"] in seen_hashes:
                reason = "exact_pixel_duplicate"
            else:
                seen_hashes.add(row["pixel_hash"])
        row["exclusion"] = reason
        row["partition"] = "excluded" if reason else ("test" if row["split"] == "test" else "development")
    return root, records, failures


def make_features(path):
    """149 simple features: regional mean/std, histogram and global summaries."""
    with Image.open(path) as im:
        image = ImageOps.pad(im.convert("L"), (64, 64), method=Image.Resampling.BILINEAR, color=0)
        pixels = np.asarray(image, dtype=np.float64) / 255.0
    # 8x8 blocks; every block contains 8x8 normalized pixels.
    blocks = pixels.reshape(8, 8, 8, 8).transpose(0, 2, 1, 3)
    means = blocks.mean(axis=(2, 3)).ravel()
    stds = blocks.std(axis=(2, 3)).ravel()
    histogram = np.histogram(pixels, bins=16, range=(0, 1))[0] / pixels.size
    summaries = [pixels.mean(), pixels.std(), np.median(pixels),
                 np.abs(np.diff(pixels, axis=0)).mean(),
                 np.abs(np.diff(pixels, axis=1)).mean()]
    return np.concatenate([means, stds, histogram, summaries])


def feature_names():
    return ([f"block_mean_r{r}_c{c}" for r in range(8) for c in range(8)]
            + [f"block_std_r{r}_c{c}" for r in range(8) for c in range(8)]
            + [f"histogram_bin_{i}" for i in range(16)]
            + ["global_mean", "global_std", "global_median", "vertical_difference", "horizontal_difference"])


def fit_scaler(x):
    """Learn only from the supplied training rows; guard constant columns."""
    mean, std = x.mean(axis=0), x.std(axis=0)
    return {"mean": mean, "std": np.where(std < 1e-12, 1.0, std)}


def scale(x, scaler):
    return (x - scaler["mean"]) / scaler["std"]


def grouped_folds(y, groups, n_folds=3, seed=SEED):
    """Greedy class-balanced group assignment, implemented without sklearn.

    Shuffle groups, then allocate largest groups first to the fold with the
    fewest samples from that group's majority class. Stable sorting makes
    ties randomized but reproducible. Stratification is approximate.
    """
    rng = np.random.default_rng(seed)
    grouped = {}
    for i, group in enumerate(groups):
        grouped.setdefault(str(group), []).append(i)
    assignment = np.full(len(y), -1, dtype=int)
    counts = np.zeros((n_folds, 2), dtype=int)
    for label in (0, 1):
        items = [(g, ids) for g, ids in grouped.items()
                 if int(y[ids].mean() >= 0.5) == label]
        rng.shuffle(items)
        items.sort(key=lambda item: len(item[1]), reverse=True)
        for group, ids in items:
            fold = int(np.argmin(counts[:, label]))
            assignment[ids] = fold
            counts[fold] += np.bincount(y[ids], minlength=2)
    folds = []
    for fold in range(n_folds):
        tr, va = np.flatnonzero(assignment != fold), np.flatnonzero(assignment == fold)
        assert not set(groups[tr]) & set(groups[va])
        assert len(np.unique(y[tr])) == len(np.unique(y[va])) == 2
        folds.append((tr, va))
    return folds, assignment


def sigmoid(z):
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


def logistic_objective(x, y, w, b, regularization, sample_weight):
    z = x @ w + b
    return float(np.mean(sample_weight * (np.logaddexp(0.0, z) - y * z))
                 + 0.5 * regularization * np.dot(w, w))


def train_logistic(x, y, regularization=0.01, max_iter=1000, learning_rate=0.2):
    """Full-batch gradient descent with backtracking; no optimizer library."""
    n, d = x.shape
    counts = np.bincount(y, minlength=2)
    sample_weight = n / (2.0 * counts[y])
    w, b = np.zeros(d), 0.0
    history = []
    for iteration in range(max_iter):
        old_loss = logistic_objective(x, y, w, b, regularization, sample_weight)
        error = sample_weight * (sigmoid(x @ w + b) - y)
        dw = x.T @ error / n + regularization * w
        db = float(error.mean())
        step = learning_rate
        for _ in range(30):
            new_w, new_b = w - step * dw, b - step * db
            loss = logistic_objective(x, y, new_w, new_b, regularization, sample_weight)
            if loss <= old_loss:
                break
            step *= 0.5
        if loss > old_loss:
            raise ArithmeticError("Gradient step failed to reduce objective")
        w, b = new_w, new_b
        history.append(loss)
        if abs(old_loss - loss) < 1e-8:
            break
    return {"kind": "logistic", "w": w, "b": b, "lambda": regularization,
            "loss_history": history, "iterations": len(history)}


def train_naive_bayes(x, y, variance_floor=1e-3):
    """Gaussian NB: class means/variances and equal class priors.

    Variance floor is in standardized squared feature units. Equal priors
    target equal class importance rather than dataset class prevalence.
    """
    means = np.stack([x[y == label].mean(axis=0) for label in (0, 1)])
    variances = np.stack([x[y == label].var(axis=0) for label in (0, 1)])
    return {"kind": "naive_bayes", "means": means,
            "variances": np.maximum(variances, variance_floor), "prior": np.array([0.5, 0.5])}


def predict_score(model, x):
    if model["kind"] == "majority":
        return np.full(len(x), model["score"], dtype=float)
    if model["kind"] == "logistic":
        return sigmoid(x @ model["w"] + model["b"])
    log_scores = []
    for label in (0, 1):
        var, mean = model["variances"][label], model["means"][label]
        log_density = -0.5 * (np.log(2 * np.pi * var) + (x - mean) ** 2 / var).sum(axis=1)
        log_scores.append(np.log(model["prior"][label]) + log_density)
    return sigmoid(log_scores[1] - log_scores[0])


def train_model(name, x, y, regularization=0.01):
    if name == "majority":
        return {"kind": "majority", "score": float(y.mean() >= 0.5)}
    if name == "gaussian_naive_bayes":
        return train_naive_bayes(x, y)
    return train_logistic(x, y, regularization=regularization)


def roc_points(y, score):
    """Threshold groups handle tied scores together; no rank-metric package."""
    order = np.argsort(-score, kind="stable")
    ys, ss = y[order], score[order]
    ends = np.r_[np.flatnonzero(ss[:-1] != ss[1:]), len(y) - 1]
    tp = np.r_[0, np.cumsum(ys)[ends]]
    fp = np.r_[0, np.cumsum(1 - ys)[ends]]
    return fp / (y == 0).sum(), tp / (y == 1).sum()


def metrics(y, score, threshold=0.5):
    prediction = (score >= threshold).astype(int)
    tp = int(((y == 1) & (prediction == 1)).sum())
    tn = int(((y == 0) & (prediction == 0)).sum())
    fp = int(((y == 0) & (prediction == 1)).sum())
    fn = int(((y == 1) & (prediction == 0)).sum())
    recall = tp / max(tp + fn, 1)
    specificity = tn / max(tn + fp, 1)
    precision = tp / max(tp + fp, 1)
    fpr, tpr = roc_points(y, score)
    auc = float(np.sum((fpr[1:] - fpr[:-1]) * (tpr[1:] + tpr[:-1]) / 2))
    return dict(accuracy=(tp + tn) / len(y), balanced_accuracy=(recall + specificity) / 2,
                precision=precision, recall=recall, specificity=specificity,
                f1=2 * precision * recall / max(precision + recall, 1e-12), roc_auc=auc,
                tn=tn, fp=fp, fn=fn, tp=tp)


def grouped_bootstrap(y, score, groups, repeats=500):
    """Percentile intervals conditional on a trained model and proxy groups."""
    rng = np.random.default_rng(SEED)
    unique = np.unique(groups)
    indices = {g: np.flatnonzero(groups == g) for g in unique}
    values = []
    for _ in range(repeats):
        sampled = rng.choice(unique, size=len(unique), replace=True)
        ids = np.concatenate([indices[g] for g in sampled])
        if len(np.unique(y[ids])) == 2:
            values.append(metrics(y[ids], score[ids]))
    return {key: [float(np.quantile([v[key] for v in values], 0.025)),
                  float(np.quantile([v[key] for v in values], 0.975))]
            for key in ("balanced_accuracy", "recall", "specificity", "f1", "roc_auc")}


def verify_math():
    """Independent mathematical checks for the hand-written ML operations."""
    y = np.array([0, 0, 1, 1])
    assert metrics(y, np.array([0.1, 0.2, 0.8, 0.9]))["roc_auc"] == 1.0
    assert metrics(y, np.array([0.9, 0.8, 0.2, 0.1]))["roc_auc"] == 0.0
    assert metrics(y, np.ones(4))["roc_auc"] == 0.5
    assert metrics(y, np.ones(4))["balanced_accuracy"] == 0.5
    toy = np.array([[1., 7.], [3., 7.], [5., 7.]])
    scaler = fit_scaler(toy)
    z = scale(toy, scaler)
    assert np.allclose(z.mean(axis=0), 0)
    assert np.isclose(z[:, 0].std(), 1) and np.allclose(z[:, 1], 0)
    # Finite differences verify the weighted logistic gradient, including penalty.
    xx = np.array([[-1., 2.], [0., 1.], [2., -1.]])
    yy = np.array([0, 1, 1])
    w, b, lam = np.array([0.2, -0.1]), 0.3, 0.1
    sw = len(yy) / (2 * np.bincount(yy)[yy])
    err = sw * (sigmoid(xx @ w + b) - yy)
    analytic = np.r_[xx.T @ err / len(yy) + lam * w, err.mean()]
    theta, numerical, eps = np.r_[w, b], [], 1e-6
    for j in range(3):
        plus, minus = theta.copy(), theta.copy()
        plus[j] += eps; minus[j] -= eps
        numerical.append((logistic_objective(xx, yy, plus[:2], plus[2], lam, sw)
                          - logistic_objective(xx, yy, minus[:2], minus[2], lam, sw)) / (2 * eps))
    assert np.allclose(analytic, numerical, atol=1e-7)
    trained = train_logistic(xx, yy, max_iter=50)
    assert np.all(np.diff(trained["loss_history"]) <= 1e-12)
    print("PASS: constant-feature scaling, tied/perfect/reversed ROC-AUC, majority baseline, finite-difference logistic gradient, monotone optimization loss.")
