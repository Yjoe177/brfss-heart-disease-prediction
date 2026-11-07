# Heart Disease Prediction from BRFSS 2015

Machine Learning Course Project 1 · Westlake University · Fall 2026

This project predicts whether a respondent reported coronary heart disease,
myocardial infarction, or angina (`_MICHD`) in the 2015 Behavioral Risk Factor
Surveillance System (BRFSS). It implements the six required course algorithms
and a NumPy-only prediction pipeline. The final model is logistic regression
trained with plain single-sample stochastic gradient descent (SGD), an L2
penalty, and a threshold chosen on training-side calibration data.

**Result:** The selected model's retrospective five-fold cross-validation mean
F1 is **0.4303 ± 0.0060** (sample standard deviation), with mean AUROC **0.8622**
and PR-AUC **0.4188**. The unregularized encoded model has a slightly higher
mean F1 (**0.4315**). These folds are descriptive rather than an untouched test:
all labeled rows had informed earlier development. Test labels were not
available, so no test-set performance is claimed. See the [report](report/report.pdf)
and [experiment protocol](experiments/README.md) for details.

## Repository layout

```text
.
├── implementations.py         # Six required course functions
├── pipeline.py                # Shared loading, preprocessing, models, metrics
├── run.py                     # Final training and prediction entry point
├── experiments/              # Reproduction and verification scripts
│   ├── README.md
│   └── results/              # Saved evidence behind the report
├── report/
│   ├── report.tex
│   └── report.pdf
├── results/
│   └── submission.csv         # Verified reference predictions
└── requirements.txt
```

The original course dataset, grader, assignment PDF, submission ZIP, LaTeX
template, and local tool configuration are not part of this repository. The
[course repository](https://github.com/LINs-lab/course_machine_learning/tree/main/projects/project1)
provides the assignment, dataset, and public tests. The included report is my
work; its LaTeX source uses the standard `IEEEtran` class from a TeX distribution.

## Run the final model

Use Python 3.9 or newer and NumPy. Install the dependency with
`python -m pip install -r requirements.txt`. The code uses only NumPy and the
Python standard library for machine learning. The coursework was checked with
Python 3.9.23 / NumPy 1.23.1 and Python 3.11.17 / NumPy 2.4.6; other versions
may differ in the last floating-point bits.

Download the official Project 1 dataset from the course repository. Put the
four supplied CSVs in `data/dataset_to_release/`:

```text
data/dataset_to_release/
├── x_train.csv
├── y_train.csv
├── x_test.csv
└── sample_submission.csv
```

Then run from the repository root:

```bash
python run.py
```

You can keep the dataset elsewhere and supply it with
`python run.py --data-dir /path/to/dataset_to_release` or the
`PROJECT1_DATA_DIR` environment variable. The output defaults to
`results/submission.csv`; `--output` chooses another path. Training takes
roughly 4–6 minutes on the CPU setup used for the course and may need several
gigabytes of RAM. An 8 GB laptop can take substantially longer because the
training design matrix alone uses about 3 GB.

The checked reference CSV has **109,379 rows** and **17,310** positive
predictions. Its MD5 is `27049fe451ef2b9b3f9b38ff230399a1`. `run.py`
checks the output header, row count, ID order, and `{-1, +1}` labels after
writing. For a fixed BLAS configuration the pipeline is deterministic, but
different NumPy/BLAS builds can change borderline scores. Compare the resulting
CSV with the included reference rather than assuming byte identity on every
machine.

## Method

1. Drop 14 administrative identifier and survey-design fields and columns
   constant in the fitting portion.
2. Fit numeric medians, means, standard deviations, and category sets on
   fitting rows only. Median-impute and standardize numeric features. One-hot
   encode features with at most 15 finite levels using `K−1` dummies and an
   explicit missing category when needed. Add an intercept.
3. Fit logistic regression for 90 epochs with a fresh seeded shuffle each
   epoch. The per-example update is
   `w ← w − ηₜ[(sigmoid(xᵢ·w) − yᵢ)xᵢ + 2λw]`, with
   `ηₜ = 0.01 / (1 + 0.1(t−1))`, seed `2027`, and `λ = 1e−5`. The penalty
   includes the intercept.
4. Apply the frozen threshold `0.21475973018819802`. It was selected on a
   52,502-row inner calibration split of the seed-42 training portion (inner
   seed 123), before the final all-data fit. This differs from the threshold
   optimized in the separate regularization study.

The project uses all 328,135 labeled rows for the final fit. The output contains
one prediction for each of the 109,379 test IDs, in the original order.

## Reproduce the evidence

With the dataset available, run the following from the repository root.
Data-reading scripts accept `--data-dir` and `PROJECT1_DATA_DIR`; see
[experiments/README.md](experiments/README.md). The five-fold run is the
expensive step (about 50–70 minutes on
the original CPU server).

```bash
python experiments/verify_sgd.py
python experiments/regularization_study.py
python experiments/fivefold_cv.py --resume
python experiments/threshold_verification.py
python experiments/feature_audit.py
python experiments/verify_results.py
```

The saved `experiments/results/` JSON, Markdown, and NPZ files preserve fold
assignments, scores, labels, threshold provenance, and verification results.
`verify_results.py` recomputes metrics and checks saved labels against
`y_train.csv`. It does **not** prove the full historical training provenance or
replace a fresh training run. Keep the NPZ files if you want to run that verifier.
The saved evidence and reference submission come from the original course run;
the full five-fold study and final fit were not rerun for this GitHub cleanup.

For the six mandatory algorithms, obtain the course's public `grading_tests`
from the linked course repository and run:

```bash
python -m pip install pytest
python -m pytest /path/to/grading_tests --submission-dir .
```

The report's regularization F1 values use thresholds optimized on the same
calibration portion used for scoring; they describe model selection rather than
independent generalization. The full-data five-fold comparison is also
retrospective because all labeled rows participated in earlier development.
