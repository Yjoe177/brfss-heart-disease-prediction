"""Reproduce the provenance of the frozen final decision threshold.

The final threshold was chosen on a training-side inner calibration split, not
on the regularization study's calibration split and not on any outer-validation
or test labels. This script reconstructs the documented split, retrains only
that inner calibration model with the frozen SGD configuration, and checks that
it reproduces the stored threshold and inner metrics.

Run:
    python experiments/threshold_verification.py
"""

import argparse
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pipeline as common

EXPECTED_THRESHOLD = 0.21475973018819802
EXPECTED_INNER_F1 = 0.43360013770548245
EXPECTED_INNER_COUNTS = {"TP": 2519, "FP": 4464, "TN": 43402, "FN": 2117}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--split-file", default=None)
    args = parser.parse_args()
    data_dir = args.data_dir or common.find_data_dir()
    output_dir = args.output_dir
    split_file = args.split_file or os.path.join(
        output_dir, "threshold_split_indices.npz")

    x, y, header = common.load_train(data_dir)
    split = np.load(split_file)
    outer_train = split["outer_train"].astype(np.int64)
    outer_val = split["outer_val"].astype(np.int64)
    inner_fit = split["inner_fit"].astype(np.int64)
    inner_cal = split["inner_cal"].astype(np.int64)
    all_rows = np.arange(y.size, dtype=np.int64)

    # Verify the saved indices against the documented split definitions.
    if not np.array_equal(np.sort(np.concatenate([outer_train, outer_val])),
                          all_rows):
        raise RuntimeError("Outer split does not cover every labeled row.")
    if np.intersect1d(outer_train, outer_val).size:
        raise RuntimeError("Outer split overlap.")
    generated_train, generated_val = common.stratified_holdout(
        y, all_rows, 0.2, 42)
    if not (np.array_equal(outer_train, generated_train) and
            np.array_equal(outer_val, generated_val)):
        raise RuntimeError("Outer split does not match seed-42 definition.")
    generated_fit, generated_cal = common.stratified_holdout(
        y, outer_train, 0.2, 123)
    if not (np.array_equal(inner_fit, generated_fit) and
            np.array_equal(inner_cal, generated_cal)):
        raise RuntimeError("Inner split does not match seed-123 definition.")

    fit_pos, fit_neg = common.class_counts(y, inner_fit)
    cal_pos, cal_neg = common.class_counts(y, inner_cal)
    candidate = common.candidate_columns(header)
    prep = common.fit_preprocessor(x, inner_fit, candidate)
    x_fit = common.make_design(x, inner_fit, prep)
    x_cal = common.make_design(x, inner_cal, prep)
    y_fit = (y[inner_fit] == 1).astype(float)
    y_cal = y[inner_cal]

    start = time.time()
    result = common.train_sgd(
        x_fit, y_fit, lr0=0.01, epochs=90, schedule="inverse",
        lam=1e-5, seed=2027)
    probabilities = common.sigmoid(x_cal @ result["w"])
    threshold, _ = common.best_threshold(probabilities, y_cal)
    at_selected = common.evaluate_scores(probabilities, y_cal, threshold)
    at_half = common.evaluate_scores(probabilities, y_cal, 0.5)
    exact = (abs(threshold - EXPECTED_THRESHOLD) < 1e-12 and
             abs(at_selected["f1"] - EXPECTED_INNER_F1) < 1e-9 and
             all(int(at_selected[key]) == EXPECTED_INNER_COUNTS[key]
                 for key in EXPECTED_INNER_COUNTS))

    # Contrast with the separate regularization study.
    with open(os.path.join(output_dir, "regularization_results.json")) as handle:
        regularization = json.load(handle)
    reg_scores = np.load(os.path.join(output_dir, "regularization_scores.npz"))
    regression_index = next(
        index for index, row in enumerate(regularization["results"])
        if abs(row["lambda"] - 1e-5) < 1e-15)
    reg_scores_l2 = reg_scores["lambda_%d_scores" % regression_index].astype(float)
    reg_labels = reg_scores["labels"].astype(int)
    reg_threshold, _ = common.best_threshold(reg_scores_l2, reg_labels)
    reg_at_optimized = common.evaluate_scores(
        reg_scores_l2, reg_labels, reg_threshold)
    reg_at_frozen = common.evaluate_scores(
        reg_scores_l2, reg_labels, EXPECTED_THRESHOLD)

    output = {
        "status": "reproduced_exactly" if exact else "not_reproduced",
        "frozen_threshold": EXPECTED_THRESHOLD,
        "reproduced_threshold": float(threshold),
        "threshold_abs_difference": abs(float(threshold) - EXPECTED_THRESHOLD),
        "expected_inner_f1": EXPECTED_INNER_F1,
        "reproduced_inner_f1": at_selected["f1"],
        "outer_split": {
            "seed": 42, "protocol": "stratified 80/20 split of all labeled rows",
            "train_rows": int(outer_train.size), "val_rows": int(outer_val.size),
            "indices_file": os.path.basename(split_file),
            "note": "Only outer_train indices are used to form the inner split; outer validation labels are not used for threshold selection."
        },
        "inner_split": {
            "seed": 123, "protocol": "stratified 80/20 split of outer_train",
            "fit_rows": int(inner_fit.size), "cal_rows": int(inner_cal.size),
            "fit_pos": fit_pos, "fit_neg": fit_neg,
            "cal_pos": cal_pos, "cal_neg": cal_neg,
        },
        "configuration": {
            "model": "L2 logistic regression",
            "optimizer": "plain single-sample SGD",
            "lr": 0.01, "schedule": "inverse",
            "schedule_formula": "eta_t = 0.01 / (1 + 0.1*(t-1))",
            "epochs": 90, "seed": 2027, "lambda": 1e-5,
            "penalty": "all weights, including intercept",
        },
        "selection_rule": (
            "maximize positive-class F1 over threshold candidates; candidates "
            "are the midpoints between adjacent sorted calibration scores; "
            "np.argmax breaks exact ties by choosing the first maximum."),
        "metrics_at_selected_threshold": at_selected,
        "metrics_at_0.5": at_half,
        "train_loss": result["data_loss"],
        "grad_inf": result["grad_inf"],
        "runtime_seconds": time.time() - start,
        "regularization_comparison": {
            "note": ("The regularization study uses the same seed-42 outer "
                     "partition for its 262,508/65,627 fit/calibration split, "
                     "but fits on all 262,508 rows and optimizes its own "
                     "threshold on its 65,627 calibration rows. It is a "
                     "different model/threshold-selection procedure."),
            "lambda": 1e-5,
            "optimized_threshold": float(reg_threshold),
            "f1_at_optimized_threshold": reg_at_optimized["f1"],
            "f1_at_frozen_threshold": reg_at_frozen["f1"],
        },
        "evidence": (
            "The archived SGD-only finalization record contained the same "
            "threshold and inner metrics. This script reproduces them from the "
            "saved split indices and documented training configuration. Only "
            "the inner calibration model is retrained; no CV, regularization "
            "training, final model, or test prediction is rerun."),
    }
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "threshold_provenance.json"), "w") as handle:
        json.dump(output, handle, indent=2)
    np.savez(os.path.join(output_dir, "threshold_inner_scores.npz"),
             labels=y_cal.astype(np.int8), probabilities=probabilities,
             threshold=float(threshold))
    print(json.dumps({
        "status": output["status"],
        "threshold": output["reproduced_threshold"],
        "inner_f1": output["reproduced_inner_f1"],
        "regularization_optimized_threshold":
            output["regularization_comparison"]["optimized_threshold"],
        "regularization_f1_at_optimized":
            output["regularization_comparison"]["f1_at_optimized_threshold"],
        "regularization_f1_at_frozen":
            output["regularization_comparison"]["f1_at_frozen_threshold"],
    }, indent=2))
    if not exact:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
