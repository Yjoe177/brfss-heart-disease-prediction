"""Shared NumPy-only data, model, and evaluation utilities for Project 1.

The final prediction script and the experiment scripts use this same data
loading, preprocessing, and SGD implementation. The module also provides
ridge regression, split generation, threshold selection, and metrics.
"""

import os
import time

import numpy as np

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz

ADMIN_COLUMNS = [
    "FMONTH", "IDATE", "IMONTH", "IDAY", "IYEAR", "DISPCODE", "SEQNO",
    "_PSU", "_STSTR", "_STRWT", "_RAWRAKE", "_WT2RAKE", "_CLLCPWT", "_LLCPWT",
]
CATEGORICAL_MAX_LEVELS = 15
READ_BLOCK_BYTES = 32 * 1024 * 1024


# ---------------------------------------------------------------------------
# Data locations and CSV loading
# ---------------------------------------------------------------------------
def find_data_dir():
    """Locate ``data/dataset_to_release`` without a machine-specific path."""
    here = os.path.dirname(os.path.abspath(__file__))
    candidates = []
    env = os.environ.get("PROJECT1_DATA_DIR")
    if env:
        candidates.append(env)
    candidates.extend([
        os.path.join(here, "data", "dataset_to_release"),
        os.path.join(os.getcwd(), "data", "dataset_to_release"),
        os.path.join(here, "..", "data", "dataset_to_release"),
        os.path.join(here, "..", "..", "data", "dataset_to_release"),
    ])
    for path in candidates:
        if os.path.isfile(os.path.join(path, "x_train.csv")):
            return os.path.normpath(path)
    raise FileNotFoundError(
        "Could not find data/dataset_to_release. Place the official CSV files "
        "there, beside run.py, or set PROJECT1_DATA_DIR.")


def count_lines(path):
    with open(path, "rb") as handle:
        return sum(1 for _ in handle)


def load_csv(path, with_ids=False):
    """Load a BRFSS CSV and drop the leading Id column from the features."""
    with open(path, "r") as handle:
        header = handle.readline().strip().split(",")
    n_rows = count_lines(path) - 1
    features = np.empty((n_rows, len(header) - 1), dtype=np.float32)
    ids = np.empty(n_rows, dtype=np.int64) if with_ids else None
    row = 0
    with open(path, "r") as handle:
        handle.readline()
        while True:
            lines = handle.readlines(READ_BLOCK_BYTES)
            if not lines:
                break
            lines = [line for line in lines if line.strip()]
            if not lines:
                continue
            block = np.genfromtxt(lines, delimiter=",", dtype=np.float64)
            if block.ndim == 1:
                block = block.reshape(1, -1)
            n = block.shape[0]
            if with_ids:
                ids[row:row + n] = block[:, 0].astype(np.int64)
            features[row:row + n] = block[:, 1:].astype(np.float32)
            row += n
    if row != n_rows:
        raise RuntimeError("Unexpected number of rows in %s" % path)
    return features, ids, header


def load_train(data_dir):
    x, _, header = load_csv(os.path.join(data_dir, "x_train.csv"))
    y = np.genfromtxt(os.path.join(data_dir, "y_train.csv"),
                      delimiter=",", skip_header=1, usecols=1, dtype=int)
    return x, y, header


def load_test(data_dir):
    return load_csv(os.path.join(data_dir, "x_test.csv"), with_ids=True)


def candidate_columns(header):
    name_to_column = {name: i - 1 for i, name in enumerate(header) if i > 0}
    administrative = np.array([name_to_column[name] for name in ADMIN_COLUMNS],
                              dtype=np.int64)
    return np.setdiff1d(np.arange(len(header) - 1, dtype=np.int64),
                        administrative)


# ---------------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------------
def detect_constant(x, rows, columns):
    constant = np.zeros(len(columns), dtype=bool)
    for k, column in enumerate(columns):
        values = x[rows, column]
        values = values[np.isfinite(values)]
        if values.size == 0 or np.all(values == values[0]):
            constant[k] = True
    return constant


def compute_stats(x, rows, columns, block=16):
    n_columns = len(columns)
    medians = np.empty(n_columns)
    means = np.empty(n_columns)
    stds = np.empty(n_columns)
    for start in range(0, n_columns, block):
        end = min(start + block, n_columns)
        values = np.asarray(x[np.ix_(rows, columns[start:end])], dtype=np.float64)
        median = np.nanmedian(values, axis=0)
        median = np.where(np.isfinite(median), median, 0.0)
        filled = np.where(np.isnan(values), median, values)
        mean = filled.mean(axis=0)
        std = filled.std(axis=0)
        std = np.where(std > 0, std, 1.0)
        medians[start:end] = median
        means[start:end] = mean
        stds[start:end] = std
    return medians, means, stds


def build_meta(x, rows, columns):
    categorical, numeric = [], []
    levels, reference, has_missing = {}, {}, {}
    for k, column in enumerate(columns):
        values = x[rows, column]
        finite = values[np.isfinite(values)]
        unique = np.unique(finite)
        if unique.size <= CATEGORICAL_MAX_LEVELS:
            categorical.append(k)
            levels[k] = unique
            uniques, counts = np.unique(finite, return_counts=True)
            reference[k] = float(uniques[int(np.argmax(counts))])
            has_missing[k] = bool(np.any(~np.isfinite(values)))
        else:
            numeric.append(k)
    n_dummies = sum(
        (levels[k].size - 1) + (1 if has_missing[k] else 0)
        for k in categorical)
    return {"categorical": categorical, "numeric": numeric, "levels": levels,
            "reference": reference, "has_missing": has_missing,
            "n_dummies": n_dummies}


def fit_preprocessor(x, rows, candidate):
    constant = detect_constant(x, rows, candidate)
    columns = np.asarray(candidate)[~constant]
    stats = compute_stats(x, rows, columns)
    meta = build_meta(x, rows, columns)
    return {"columns": columns, "stats": stats, "meta": meta}


def make_design(x, rows, prep):
    medians, means, stds = prep["stats"]
    meta = prep["meta"]
    dimension = 1 + len(meta["numeric"]) + meta["n_dummies"]
    design = np.zeros((len(rows), dimension), dtype=np.float64)
    design[:, 0] = 1.0
    position = 1
    numeric = np.asarray(meta["numeric"], dtype=np.int64)
    for start in range(0, numeric.size, 16):
        end = min(start + 16, numeric.size)
        indices = numeric[start:end]
        values = np.asarray(x[np.ix_(rows, prep["columns"][indices])],
                            dtype=np.float64)
        values = np.where(np.isnan(values), medians[indices], values)
        values = (values - means[indices]) / stds[indices]
        design[:, position:position + indices.size] = values
        position += indices.size
    for k in meta["categorical"]:
        values = x[rows, prep["columns"][k]]
        reference = meta["reference"][k]
        for level in meta["levels"][k]:
            if level == reference:
                continue
            design[:, position] = (values == level)
            position += 1
        if meta["has_missing"][k]:
            design[:, position] = ~np.isfinite(values)
            position += 1
    if position != dimension:
        raise RuntimeError("Design matrix dimension mismatch.")
    return design


def fit_numeric_preprocessor(x, rows, candidate):
    constant = detect_constant(x, rows, candidate)
    columns = np.asarray(candidate)[~constant]
    stats = compute_stats(x, rows, columns)
    return {"columns": columns, "stats": stats}


def make_numeric_design(x, rows, prep):
    medians, means, stds = prep["stats"]
    columns = prep["columns"]
    design = np.zeros((len(rows), 1 + len(columns)), dtype=np.float64)
    design[:, 0] = 1.0
    for start in range(0, len(columns), 16):
        end = min(start + 16, len(columns))
        indices = np.arange(start, end)
        values = np.asarray(x[np.ix_(rows, columns[indices])], dtype=np.float64)
        values = np.where(np.isnan(values), medians[indices], values)
        values = (values - means[indices]) / stds[indices]
        design[:, 1 + start:1 + end] = values
    return design


def encoded_summary(prep, design_dimension):
    meta = prep["meta"]
    return {"n_input_features": int(len(prep["columns"])),
            "n_numeric": int(len(meta["numeric"])),
            "n_categorical": int(len(meta["categorical"])),
            "n_dummies": int(meta["n_dummies"]),
            "design_dimension": int(design_dimension)}


def numeric_summary(prep, design_dimension):
    return {"n_input_features": int(len(prep["columns"])),
            "n_numeric": int(len(prep["columns"])),
            "n_categorical": 0, "n_dummies": 0,
            "design_dimension": int(design_dimension)}


# ---------------------------------------------------------------------------
# SGD for logistic regression
# ---------------------------------------------------------------------------
def sigmoid(z):
    z = np.asarray(z, dtype=float)
    output = np.empty_like(z)
    positive = z >= 0
    output[positive] = 1.0 / (1.0 + np.exp(-z[positive]))
    exp_z = np.exp(z[~positive])
    output[~positive] = exp_z / (1.0 + exp_z)
    return output


def sigmoid_scalar(z):
    if z >= 0.0:
        return 1.0 / (1.0 + np.exp(-z))
    exp_z = np.exp(z)
    return exp_z / (1.0 + exp_z)


def learning_rate(epoch, lr0, decay):
    """Epoch-indexed inverse schedule used by the final model."""
    return lr0 / (1.0 + decay * (epoch - 1))


def logistic_loss(x, y, w):
    logits = x @ w
    return float(np.mean(np.logaddexp(0.0, logits) - y * logits))


def logistic_grad(x, y, w, lam):
    n = x.shape[0]
    grad = x.T @ (sigmoid(x @ w) - y) / n
    if lam:
        grad = grad + 2.0 * lam * w
    return grad


def train_sgd(x, y, lr0=0.01, epochs=90, schedule="inverse", lam=0.0,
              seed=2027, decay=0.1, eval_every=15, progress=False):
    """Plain single-sample SGD for L2-regularised logistic regression.

    Objective: mean logistic loss + lam * ||w||^2, with the penalty applied to
    all weights including the intercept. The update is
        w <- w - eta_t * ((sigmoid(x_i @ w) - y_i) * x_i + 2 * lam * w).
    """
    n, d = x.shape
    w = np.zeros(d)
    rng = np.random.default_rng(seed)
    history = []
    start = time.time()
    for epoch in range(1, epochs + 1):
        eta = (learning_rate(epoch, lr0, decay)
               if schedule == "inverse" else lr0)
        order = rng.permutation(n)
        for i in order:
            row = x[i]
            probability = sigmoid_scalar(float(row @ w))
            gradient = (probability - y[i]) * row
            if lam:
                gradient = gradient + 2.0 * lam * w
            w -= eta * gradient
        if epoch % eval_every == 0 or epoch == epochs:
            data_loss = logistic_loss(x, y, w)
            penalty = lam * float(np.sum(w * w))
            history.append({"epoch": epoch, "eta": eta,
                            "data_loss": data_loss,
                            "penalized_loss": data_loss + penalty})
            if progress:
                print("epoch %d/%d  loss %.8f  elapsed %.1fs" % (
                    epoch, epochs, data_loss, time.time() - start), flush=True)
    data_loss = logistic_loss(x, y, w)
    penalty = lam * float(np.sum(w * w))
    grad = logistic_grad(x, y, w, lam)
    return {
        "w": w, "data_loss": data_loss, "penalty": penalty,
        "penalized_loss": data_loss + penalty,
        "grad_inf": float(np.max(np.abs(grad))),
        "grad_l2": float(np.linalg.norm(grad)),
        "history": history,
        "runtime": time.time() - start,
    }


def fit_linear(x, y, lam):
    """Ridge least squares via normal equations, penalty on all weights."""
    n, d = x.shape
    matrix = x.T @ x + 2.0 * n * lam * np.eye(d)
    target = x.T @ y
    try:
        return np.linalg.solve(matrix, target)
    except np.linalg.LinAlgError:
        return np.linalg.pinv(matrix) @ target


# ---------------------------------------------------------------------------
# Splits
# ---------------------------------------------------------------------------
def stratified_holdout(y, rows, val_fraction, seed):
    """Split rows into fitting and calibration parts, stratified by label."""
    rng = np.random.default_rng(seed)
    positive = rows[y[rows] == 1].copy()
    negative = rows[y[rows] == -1].copy()
    rng.shuffle(positive)
    rng.shuffle(negative)
    n_val_pos = int(round(len(positive) * val_fraction))
    n_val_neg = int(round(len(negative) * val_fraction))
    cal = np.sort(np.concatenate([positive[:n_val_pos], negative[:n_val_neg]]))
    fit = np.sort(np.concatenate([positive[n_val_pos:], negative[n_val_neg:]]))
    return fit, cal


def stratified_folds(y, rows, n_folds, seed):
    """Exact stratified fold assignment; no row is used in two folds."""
    rng = np.random.default_rng(seed)
    positive = rows[y[rows] == 1].copy()
    negative = rows[y[rows] == -1].copy()
    rng.shuffle(positive)
    rng.shuffle(negative)
    positive_folds = np.array_split(positive, n_folds)
    negative_folds = np.array_split(negative, n_folds)
    folds = [np.sort(np.concatenate([positive_folds[k], negative_folds[k]]))
             for k in range(n_folds)]
    union = np.sort(np.concatenate(folds))
    if not np.array_equal(union, np.sort(rows)):
        raise RuntimeError("Fold union does not equal the requested rows.")
    for i in range(n_folds):
        for j in range(i + 1, n_folds):
            if np.intersect1d(folds[i], folds[j]).size:
                raise RuntimeError("Fold overlap detected.")
    return folds


def class_counts(y, rows):
    positive = int(np.sum(y[rows] == 1))
    negative = int(np.sum(y[rows] == -1))
    return positive, negative


# ---------------------------------------------------------------------------
# Metrics
# ---------------------------------------------------------------------------
def safe_div(numerator, denominator):
    return float(numerator) / float(denominator) if denominator else 0.0


def metrics_at(scores, y, threshold):
    prediction = np.where(scores >= threshold, 1, -1)
    tp = int(np.sum((prediction == 1) & (y == 1)))
    fp = int(np.sum((prediction == 1) & (y == -1)))
    tn = int(np.sum((prediction == -1) & (y == -1)))
    fn = int(np.sum((prediction == -1) & (y == 1)))
    precision = safe_div(tp, tp + fp)
    recall = safe_div(tp, tp + fn)
    specificity = safe_div(tn, tn + fp)
    return {
        "threshold": float(threshold), "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": safe_div(tp + tn, len(y)), "precision": precision,
        "recall": recall,
        "f1": safe_div(2.0 * precision * recall, precision + recall),
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
    }


def best_threshold(scores, y):
    order = np.argsort(scores)[::-1]
    positive = (y[order] == 1).astype(float)
    tp = np.cumsum(positive)
    fp = np.cumsum(1.0 - positive)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positive.sum()
    f1 = 2.0 * precision * recall / np.maximum(precision + recall, 1e-12)
    index = int(np.argmax(f1))
    if index + 1 < len(order):
        threshold = 0.5 * (scores[order[index]] + scores[order[index + 1]])
    else:
        threshold = scores[order[index]] - 1e-12
    return float(threshold), float(f1[index])


def roc_auc(scores, y):
    order = np.argsort(scores)[::-1]
    positive = (y[order] == 1)
    tpr = np.concatenate([[0.0], np.cumsum(positive) / positive.sum()])
    fpr = np.concatenate([[0.0], np.cumsum(~positive) / (~positive).sum()])
    return float(_trapz(tpr, fpr))


def pr_auc(scores, y):
    order = np.argsort(scores)[::-1]
    positive = (y[order] == 1)
    tp = np.cumsum(positive)
    fp = np.cumsum(~positive)
    precision = tp / np.maximum(tp + fp, 1.0)
    recall = tp / positive.sum()
    precision = np.concatenate([[1.0], precision, [precision[-1]]])
    recall = np.concatenate([[0.0], recall, [1.0]])
    return float(_trapz(precision, recall))


def evaluate_scores(scores, y, threshold):
    metrics = metrics_at(scores, y, threshold)
    metrics["roc_auc"] = roc_auc(scores, y)
    metrics["pr_auc"] = pr_auc(scores, y)
    return metrics
