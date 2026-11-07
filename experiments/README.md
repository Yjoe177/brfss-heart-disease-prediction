# SGD experiments

This directory reproduces the experimental evidence in the report. All machine
learning computations use NumPy and the Python standard library only.

## Data

Place the official files exactly as supplied:

```text
data/dataset_to_release/
├── x_train.csv
├── y_train.csv
└── x_test.csv
```

Data-reading scripts search for `data/dataset_to_release` beside `run.py` or
in the current directory, or use `PROJECT1_DATA_DIR`. They also accept
`--data-dir`. No machine-specific path is required.

## Workflow

Run from the project root (the directory containing `run.py`):

```bash
python experiments/verify_sgd.py
python experiments/regularization_study.py
python experiments/fivefold_cv.py --resume
python experiments/threshold_verification.py
python experiments/feature_audit.py
python experiments/verify_results.py
```

Windows PowerShell equivalents:

```powershell
python experiments\verify_sgd.py
python experiments\regularization_study.py
python experiments\fivefold_cv.py --resume
python experiments\threshold_verification.py
python experiments\feature_audit.py
python experiments\verify_results.py
```

### `verify_sgd.py`

Checks the analytic logistic gradient against central finite differences for
the unregularized and L2 objectives, the average of per-sample gradients against
the full gradient, a manual single-example update, the epoch-indexed inverse
learning-rate schedule, deterministic shuffling, finite training behaviour, and
the actual one-sample training update. Output: `results/verify_sgd.json`.

### `regularization_study.py`

Uses the seed-42 stratified 80/20 partition of all 328,135 labeled examples:
262,508 fitting and 65,627 calibration rows. This is the same partition used
during earlier development. The fitting portion trains each L2 setting; the
calibration portion selects the threshold and supplies the reported metrics,
so those F1 values describe model selection and regularization behavior, not
independent generalization. The settings are `lambda = 0, 1e-5, 1e-4, 1e-3`.
This is a separate training-side study, not the five-fold CV.

Outputs:

```text
results/regularization_results.json
results/regularization_scores.npz
results/regularization_weights.npz
results/regularization_summary.md
```

### `fivefold_cv.py`

Runs stratified five-fold cross-validation on **all 328,135 labeled examples**
(fold seed 2028). Every fold contains:

- 65,627 held-out rows;
- 5,795 held-out positives;
- 59,832 held-out negatives;
- 262,508 training rows.

For each fold, a stratified 80/20 inner split of the fold-training portion is
used to choose the threshold. Preprocessing is fitted only on the relevant
training portion. The held-out fold never influences its own preprocessing,
threshold, or model fit. The outer model is then refit on the complete
fold-training portion with the frozen threshold.

Models compared on the same five folds:

1. always-negative baseline;
2. ridge regression with one-hot features (normal equations for the objective
   `0.5 * mean((X @ w - y)**2) + lambda * sum(w**2)`);
3. logistic regression with numeric survey codes, trained by plain SGD;
4. unregularized logistic regression with one-hot features, trained by plain
   SGD;
5. L2 logistic regression with one-hot features, trained by plain SGD
   (`lambda = 1e-5`).

The five-fold run is a retrospective evaluation of a frozen procedure: all
labeled rows informed earlier development, and the folds are correlated.

Outputs:

```text
results/folds.npz
results/fivefold_results.json
results/fivefold_scores.npz
results/fivefold_summary.md
```

`fivefold_scores.npz` stores the held-out labels and scores for every fold and
model, so AUROC and PR-AUC can be recomputed independently. The `.work/`
directory contains resumable per-fold checkpoints; it is not part of the final
results and can be deleted after a successful run.

### `verify_results.py`

Loads the saved fold assignments, raw held-out scores, aggregate JSON,
regularization outputs, threshold-provenance evidence, and feature audit. It
independently recomputes confusion matrices, F1, precision, recall, accuracy,
balanced accuracy, AUROC, PR-AUC, mean F1, sample SD, and paired fold
differences. It verifies saved fold labels against the genuine `y_train.csv`,
the regularization and threshold-calibration labels against `y_train.csv` at
their exact split indices, the corrected ridge objective, Markdown-table
agreement, threshold provenance, and the header-level feature audit. Output:
`results/verification_report.json`.

The verifier distinguishes raw-array recomputation, targeted inner-calibration
reruns, source inspection, and claims that would require a full training rerun
or independent test data. It does not claim to prove complete historical
training provenance.

### `threshold_verification.py`

Reconstructs the training-side inner split used for the frozen final threshold:
seed-42 outer split, seed-123 inner 80/20 split of the 262,508 training rows
(210,006 fit / 52,502 calibration). It retrains only that inner calibration
model with the frozen SGD configuration and checks whether it reproduces the
threshold `0.21475973018819802` and inner F1 `0.43360013770548245`. It also
contrasts the frozen threshold with the separate regularization study's own
optimized threshold. Outputs:

```text
results/threshold_provenance.json
results/threshold_inner_scores.npz
results/threshold_split_indices.npz
```

### `feature_audit.py`

Reads only the official CSV headers and checks that `_MICHD` is the label, not a
feature; that `CVDINFR4` and `CVDCRHD4` are absent; and that `CVDSTRK3` is
present. Output: `results/feature_audit.json`. This is a header-level audit, not
proof that no derived relationship exists.

## Configuration

The final model and the CV comparison use:

- plain single-sample SGD, no momentum or adaptive scaling;
- initial learning rate `0.01`;
- inverse epoch schedule `eta_t = 0.01 / (1 + 0.1(t-1))`;
- 90 epochs;
- seed 2027;
- L2 penalty on all weights, including the intercept;
- final L2 strength `1e-5`;
- ridge regularization `1e-4` for the comparison model.

The hyperparameters and final threshold were frozen before the full-data CV.
Because the labeled data had informed earlier training-side choices, the
full-data CV is a retrospective evaluation of a frozen procedure, not an
untouched independent test. No held-out fold is used to tune a hyperparameter
for that fold.

## Runtime and memory

- `verify_sgd.py`: under one minute.
- `regularization_study.py`: about 15–20 minutes on a CPU server, several GB of RAM.
- `fivefold_cv.py`: about 50–70 minutes on a CPU server, roughly 6–8 GB of RAM.
- `threshold_verification.py`: about 2–3 minutes; it retrains only the inner
  calibration model.
- `feature_audit.py`: under one minute.
- `verify_results.py`: under one minute.

Exact byte-level metrics can vary in the last floating-point digits across
different BLAS builds. The saved JSON and NPZ files are the reference outputs.
