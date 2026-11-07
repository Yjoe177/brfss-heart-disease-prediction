"""Verify the saved SGD, regularization and five-fold results.

Run:
    python experiments/verify_results.py

The script imports ``pipeline`` for CSV loading, documented split
reconstruction, and a synthetic ridge-solver identity check. It does not call
the training routine for the saved experiments; metric and threshold arithmetic
is reimplemented independently here. It checks fold coverage, genuine-label
provenance, summary statistics, paired differences, ridge bookkeeping, threshold
provenance, Markdown tables, and the header-level feature audit.
"""

import argparse
import json
import os
import re
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pipeline as common

_trapz = np.trapezoid if hasattr(np, "trapezoid") else np.trapz


def metrics_from_scores(scores, labels, threshold):
    prediction = np.where(scores >= threshold, 1, -1)
    tp = int(np.sum((prediction == 1) & (labels == 1)))
    fp = int(np.sum((prediction == 1) & (labels == -1)))
    tn = int(np.sum((prediction == -1) & (labels == -1)))
    fn = int(np.sum((prediction == -1) & (labels == 1)))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    specificity = tn / (tn + fp) if tn + fp else 0.0
    accuracy = (tp + tn) / len(labels)

    order = np.argsort(scores)[::-1]
    positive = labels[order] == 1
    tpr = np.concatenate([[0.0], np.cumsum(positive) / positive.sum()])
    fpr = np.concatenate([[0.0], np.cumsum(~positive) / (~positive).sum()])
    auc = float(_trapz(tpr, fpr))

    order = np.argsort(scores)[::-1]
    positive = labels[order] == 1
    true_positive = np.cumsum(positive)
    false_positive = np.cumsum(~positive)
    precision_curve = true_positive / np.maximum(true_positive + false_positive, 1)
    recall_curve = true_positive / positive.sum()
    precision_curve = np.concatenate([[1.0], precision_curve, [precision_curve[-1]]])
    recall_curve = np.concatenate([[0.0], recall_curve, [1.0]])
    pr_auc = float(_trapz(precision_curve, recall_curve))
    return {
        "TP": tp, "FP": fp, "TN": tn, "FN": fn,
        "accuracy": accuracy, "precision": precision, "recall": recall,
        "f1": 2 * precision * recall / (precision + recall)
        if precision + recall else 0.0,
        "specificity": specificity,
        "balanced_accuracy": 0.5 * (recall + specificity),
        "roc_auc": auc, "pr_auc": pr_auc,
    }


def close(actual, expected, tolerance=1e-12):
    return bool(abs(float(actual) - float(expected)) <= tolerance)


def md_close(actual, expected):
    """Compare a rendered Markdown value (4 or 3 decimals)."""
    return close(actual, expected, tolerance=1e-3)


def best_threshold_independent(scores, labels):
    """Independent reconstruction of the midpoint threshold rule."""
    order = np.argsort(scores)[::-1]
    positive = (labels[order] == 1).astype(float)
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


def parse_markdown_table(path):
    rows = {}
    with open(path, "r") as handle:
        for line in handle:
            line = line.strip()
            if not line.startswith("|"):
                continue
            cells = [cell.strip() for cell in line.strip("|").split("|")]
            if len(cells) < 2:
                continue
            if cells[0].lower() in ("model", "λ", "lambda") or set(
                    cells[0]) <= set("-: "):
                continue
            rows[cells[0]] = cells
    return rows


def check_metric_dict(actual, expected, context, failures):
    for key in ("TP", "FP", "TN", "FN"):
        if int(actual[key]) != int(expected[key]):
            failures.append("%s: %s %s != %s" % (
                context, key, actual[key], expected[key]))
    for key in ("accuracy", "precision", "recall", "f1", "specificity",
                "balanced_accuracy", "roc_auc", "pr_auc"):
        if not close(actual[key], expected[key]):
            failures.append("%s: %s %.15g != %.15g" % (
                context, key, actual[key], expected[key]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=os.path.join(HERE, "results"))
    args = parser.parse_args()
    data_dir = args.data_dir or common.find_data_dir()
    output_dir = args.output_dir
    failures = []

    # SGD mathematical checks.
    with open(os.path.join(output_dir, "verify_sgd.json")) as handle:
        sgd = json.load(handle)
    if not sgd.get("passed"):
        failures.append("verify_sgd.json does not report passed=true")

    # Full-data fold coverage.
    x, y, header = common.load_train(data_dir)
    all_rows = np.arange(y.size, dtype=np.int64)
    folds_file = np.load(os.path.join(output_dir, "folds.npz"))
    folds = [folds_file["fold%d" % k].astype(np.int64) for k in range(5)]
    if y.size != 328135:
        failures.append("dataset rows %d != 328135" % y.size)
    union = np.sort(np.concatenate(folds))
    if not np.array_equal(union, all_rows):
        failures.append("fold union does not cover every labeled row")
    for k, fold in enumerate(folds):
        pos = int(np.sum(y[fold] == 1))
        neg = int(np.sum(y[fold] == -1))
        if fold.size != 65627 or pos != 5795 or neg != 59832:
            failures.append("fold %d counts invalid: %d/%d/%d" % (
                k, fold.size, pos, neg))
        for other in range(k + 1, 5):
            if np.intersect1d(fold, folds[other]).size:
                failures.append("fold %d overlaps fold %d" % (k, other))
        train = np.setdiff1d(all_rows, fold)
        if train.size != 262508:
            failures.append("fold %d training size %d != 262508" % (k, train.size))

    with open(os.path.join(output_dir, "fivefold_results.json")) as handle:
        cv = json.load(handle)
    scores_file = np.load(os.path.join(output_dir, "fivefold_scores.npz"))
    model_names = ["ridge", "logistic_numeric", "logistic_encoded",
                   "reg_logistic_encoded"]
    for fold_index, fold_record in enumerate(cv["folds"]):
        labels = scores_file["fold%d_labels" % fold_index].astype(int)
        if labels.size != 65627:
            failures.append("fold %d raw label size invalid" % fold_index)
        expected_labels = y[folds[fold_index]].astype(int)
        if not np.array_equal(labels, expected_labels):
            failures.append("fold %d saved labels do not match y_train at fold indices"
                            % fold_index)
        for model in model_names:
            stored = fold_record["models"][model]["metrics"]
            scores = scores_file["fold%d_%s_scores" % (fold_index, model)].astype(float)
            threshold = fold_record["models"][model]["threshold"]
            recomputed = metrics_from_scores(scores, labels, threshold)
            check_metric_dict(recomputed, stored,
                              "fold %d %s" % (fold_index, model), failures)
        majority = fold_record["models"]["majority"]["metrics"]
        prediction = scores_file["fold%d_majority_prediction" % fold_index]
        if not np.all(prediction == -1):
            failures.append("fold %d majority prediction not all -1" % fold_index)
        if not close(majority["f1"], 0.0) or not close(majority["roc_auc"], 0.5):
            failures.append("fold %d majority metrics invalid" % fold_index)

    # Ridge bookkeeping and normal-equation objective identity.
    ridge_lambda = float(cv["metadata"]["ridge_lambda"])
    for fold_record in cv["folds"]:
        ridge = fold_record["models"]["ridge"]
        expected_half = 0.5 * float(ridge["train_mse"])
        expected_penalized = expected_half + ridge_lambda * float(ridge["weight_l2"]) ** 2
        if not close(ridge.get("half_train_mse", expected_half), expected_half):
            failures.append("fold %d ridge half-MSE mismatch" % fold_record["fold"])
        if not close(ridge["penalized_train_objective"], expected_penalized):
            failures.append("fold %d ridge objective does not match 0.5*MSE + lambda*||w||^2"
                            % fold_record["fold"])
    rng = np.random.default_rng(0)
    x_small = rng.normal(size=(40, 5))
    x_small[:, 0] = 1.0
    y_small = (rng.random(40) > 0.5).astype(float)
    small_lambda = 1e-4
    w_small = common.fit_linear(x_small, y_small, small_lambda)
    ridge_gradient = (x_small.T @ (x_small @ w_small - y_small) / 40.0 +
                      2.0 * small_lambda * w_small)
    ridge_identity_max_gradient = float(np.max(np.abs(ridge_gradient)))
    if ridge_identity_max_gradient > 1e-8:
        failures.append("ridge normal equations do not match the corrected objective")

    # Aggregate mean/SD and paired differences.
    summary = {}
    for model in ["majority"] + model_names:
        f1 = np.array([record["models"][model]["metrics"]["f1"]
                       for record in cv["folds"]])
        auc = np.array([record["models"][model]["metrics"]["roc_auc"]
                        for record in cv["folds"]])
        pr = np.array([record["models"][model]["metrics"]["pr_auc"]
                       for record in cv["folds"]])
        summary[model] = {"f1_mean": float(f1.mean()),
                          "f1_sd": float(f1.std(ddof=1)),
                          "auc_mean": float(auc.mean()),
                          "pr_mean": float(pr.mean())}
        stored = cv["summary"][model]
        for key in ("f1_mean", "f1_sd", "auc_mean", "pr_mean"):
            if not close(summary[model][key], stored[key]):
                failures.append("summary %s %s mismatch" % (model, key))

    def paired(a, b):
        values = np.array([
            record["models"][a]["metrics"]["f1"] -
            record["models"][b]["metrics"]["f1"] for record in cv["folds"]])
        return float(values.mean()), float(values.std(ddof=1)), values.tolist()

    for key, (a, b) in {
            "reg_logistic_minus_unregularized": ("reg_logistic_encoded", "logistic_encoded"),
            "encoded_minus_numeric": ("logistic_encoded", "logistic_numeric"),
            "unregularized_minus_ridge": ("logistic_encoded", "ridge")}.items():
        mean, sd, values = paired(a, b)
        stored = cv["paired_differences"][key]
        if not close(mean, stored["mean"]) or not close(sd, stored["sample_sd"]):
            failures.append("paired %s mismatch" % key)
        if not np.allclose(values, stored["values"]):
            failures.append("paired %s values mismatch" % key)

    # Regularization outputs.
    with open(os.path.join(output_dir, "regularization_results.json")) as handle:
        regularization = json.load(handle)
    regularization_scores = np.load(os.path.join(output_dir, "regularization_scores.npz"))
    regularization_weights = np.load(os.path.join(output_dir, "regularization_weights.npz"))
    labels = regularization_scores["labels"].astype(int)
    regularization_fit, regularization_cal = common.stratified_holdout(
        y, all_rows, 0.2, 42)
    regularization_expected_labels = y[regularization_cal].astype(int)
    regularization_labels_match = bool(np.array_equal(
        labels, regularization_expected_labels))
    if not regularization_labels_match:
        failures.append(
            "regularization_scores labels do not match y_train at the seed-42 "
            "calibration indices")
    if (regularization_cal.size != 65627 or
            int(regularization["metadata"]["fit_rows"]) != regularization_fit.size or
            int(regularization["metadata"]["cal_rows"]) != regularization_cal.size):
        failures.append("regularization split counts do not match metadata")
    for index, record in enumerate(regularization["results"]):
        scores = regularization_scores["lambda_%d_scores" % index].astype(float)
        recomputed = metrics_from_scores(scores, labels, record["threshold"])
        check_metric_dict(recomputed, record["metrics"],
                          "regularization lambda %g" % record["lambda"], failures)
        weights = regularization_weights["lambda_%d_weights" % index]
        if not close(record["weight_l2"], np.linalg.norm(weights)):
            failures.append("regularization lambda %g weight norm mismatch" % record["lambda"])
        if not np.isfinite(record["calibration_loss"]):
            failures.append("regularization lambda %g non-finite calibration loss" % record["lambda"])
        if not close(record["train_calibration_gap"],
                     record["calibration_loss"] - record["train_loss"]):
            failures.append("regularization lambda %g gap mismatch" % record["lambda"])

    # Markdown summaries must agree with the saved JSON.
    markdown_checks = {}
    fivefold_md = parse_markdown_table(
        os.path.join(output_dir, "fivefold_summary.md"))
    for model in ["majority"] + model_names:
        row = fivefold_md.get(model)
        ok = row is not None and len(row) >= 4
        if ok:
            try:
                f1_parts = [float(part.strip()) for part in row[1].split("+/-")]
                ok = (len(f1_parts) == 2 and
                      md_close(f1_parts[0], summary[model]["f1_mean"]) and
                      md_close(f1_parts[1], summary[model]["f1_sd"]) and
                      md_close(float(row[2]), summary[model]["auc_mean"]) and
                      md_close(float(row[3]), summary[model]["pr_mean"]))
            except ValueError:
                ok = False
        markdown_checks["fivefold_" + model] = bool(ok)
        if not ok:
            failures.append("fivefold_summary.md row mismatch for %s" % model)

    with open(os.path.join(output_dir, "fivefold_summary.md"), "r") as handle:
        fivefold_text = handle.read()
    for key, (a, b) in {
            "reg_logistic_minus_unregularized": ("reg_logistic_encoded", "logistic_encoded"),
            "encoded_minus_numeric": ("logistic_encoded", "logistic_numeric"),
            "unregularized_minus_ridge": ("logistic_encoded", "ridge")}.items():
        stored = cv["paired_differences"][key]
        expected_line = "- F1(%s) - F1(%s): %+.4f +/- %.4f" % (
            a, b, stored["mean"], stored["sample_sd"])
        ok = expected_line in fivefold_text
        markdown_checks["paired_" + key] = bool(ok)
        if not ok:
            failures.append("fivefold_summary.md paired line mismatch for %s" % key)

    regularization_md = parse_markdown_table(
        os.path.join(output_dir, "regularization_summary.md"))
    for record in regularization["results"]:
        row = None
        for key, cells in regularization_md.items():
            try:
                if abs(float(key) - float(record["lambda"])) < 1e-15:
                    row = cells
                    break
            except ValueError:
                continue
        ok = row is not None and len(row) >= 9
        if ok:
            try:
                ok = (md_close(float(row[1]), record["metrics"]["f1"]) and
                      md_close(float(row[2]), record["metrics"]["precision"]) and
                      md_close(float(row[3]), record["metrics"]["recall"]) and
                      md_close(float(row[4]), record["metrics"]["roc_auc"]) and
                      md_close(float(row[5]), record["metrics"]["pr_auc"]) and
                      md_close(float(row[6]), record["train_loss"]) and
                      md_close(float(row[7]), record["calibration_loss"]) and
                      md_close(float(row[8]), record["weight_l2"]))
            except ValueError:
                ok = False
        markdown_checks["regularization_lambda_%g" % record["lambda"]] = bool(ok)
        if not ok:
            failures.append("regularization_summary.md row mismatch for lambda %g"
                            % record["lambda"])

    # Threshold provenance evidence.
    threshold_checks = {}
    with open(os.path.join(output_dir, "threshold_provenance.json")) as handle:
        provenance = json.load(handle)
    expected_threshold = 0.21475973018819802
    threshold_checks["status_reproduced_exactly"] = (
        provenance.get("status") == "reproduced_exactly")
    threshold_checks["threshold_value"] = close(
        provenance.get("frozen_threshold", -1), expected_threshold)
    threshold_checks["reproduced_threshold"] = close(
        provenance.get("reproduced_threshold", -1), expected_threshold)
    if not all(threshold_checks.values()):
        failures.append("threshold provenance status/value mismatch")

    threshold_split = np.load(
        os.path.join(output_dir, "threshold_split_indices.npz"))
    outer_train = threshold_split["outer_train"].astype(np.int64)
    outer_val = threshold_split["outer_val"].astype(np.int64)
    inner_fit = threshold_split["inner_fit"].astype(np.int64)
    inner_cal = threshold_split["inner_cal"].astype(np.int64)
    generated_train, generated_val = common.stratified_holdout(
        y, all_rows, 0.2, 42)
    generated_fit, generated_cal = common.stratified_holdout(
        y, outer_train, 0.2, 123)
    threshold_checks["outer_split_matches_seed42"] = bool(
        np.array_equal(outer_train, generated_train) and
        np.array_equal(outer_val, generated_val))
    threshold_checks["inner_split_matches_seed123"] = bool(
        np.array_equal(inner_fit, generated_fit) and
        np.array_equal(inner_cal, generated_cal))
    threshold_checks["inner_counts"] = bool(
        inner_fit.size == 210006 and inner_cal.size == 52502 and
        int(np.sum(y[inner_fit] == 1)) == 18544 and
        int(np.sum(y[inner_cal] == 1)) == 4636)
    if not all(threshold_checks.values()):
        failures.append("threshold split indices do not match documented seeds")

    inner = np.load(os.path.join(output_dir, "threshold_inner_scores.npz"))
    inner_labels = inner["labels"].astype(int)
    inner_expected_labels = y[inner_cal].astype(int)
    threshold_labels_match = bool(np.array_equal(
        inner_labels, inner_expected_labels))
    if not threshold_labels_match:
        failures.append(
            "threshold_inner_scores labels do not match y_train at the seed-123 "
            "inner-calibration indices")
    if inner_labels.size != 52502:
        failures.append("threshold inner label count is not 52,502")
    inner_probabilities = inner["probabilities"].astype(float)
    independent_threshold, _ = best_threshold_independent(
        inner_probabilities, inner_labels)
    threshold_checks["inner_threshold_recomputed"] = close(
        independent_threshold, expected_threshold)
    independent_metrics = metrics_from_scores(
        inner_probabilities, inner_labels, expected_threshold)
    check_metric_dict(independent_metrics,
                      provenance["metrics_at_selected_threshold"],
                      "threshold inner calibration", failures)
    threshold_checks["inner_metrics_recomputed"] = bool(
        close(independent_metrics["f1"],
              provenance["metrics_at_selected_threshold"]["f1"]))

    run_path = os.path.join(HERE, "..", "run.py")
    with open(run_path, "r") as handle:
        run_source = handle.read()
    match = re.search(r"DECISION_THRESHOLD\s*=\s*([0-9.eE+-]+)",
                      run_source)
    threshold_checks["run_py_threshold_constant"] = bool(
        match is not None and close(float(match.group(1)), expected_threshold))
    if not all(threshold_checks.values()):
        failures.append("run.py threshold or provenance evidence mismatch")

    regression = provenance["regularization_comparison"]
    reg_l2_index = next(
        index for index, row in enumerate(regularization["results"])
        if abs(row["lambda"] - 1e-5) < 1e-15)
    reg_l2_scores = regularization_scores[
        "lambda_%d_scores" % reg_l2_index].astype(float)
    reg_l2_threshold, _ = best_threshold_independent(reg_l2_scores, labels)
    reg_l2_optimized_f1 = metrics_from_scores(
        reg_l2_scores, labels, reg_l2_threshold)["f1"]
    reg_l2_frozen_f1 = metrics_from_scores(
        reg_l2_scores, labels, expected_threshold)["f1"]
    threshold_checks["regularization_optimized_threshold"] = close(
        reg_l2_threshold, regression["optimized_threshold"])
    threshold_checks["regularization_f1_at_optimized"] = close(
        reg_l2_optimized_f1, regression["f1_at_optimized_threshold"])
    threshold_checks["regularization_f1_at_frozen"] = close(
        reg_l2_frozen_f1, regression["f1_at_frozen_threshold"])
    if not all(threshold_checks.values()):
        failures.append("regularization/frozen threshold comparison mismatch")

    # Header-level feature audit.
    feature_audit_path = os.path.join(output_dir, "feature_audit.json")
    feature_audit = {}
    if os.path.exists(feature_audit_path):
        with open(feature_audit_path) as handle:
            feature_audit = json.load(handle)
    feature_ok = bool(feature_audit.get("passed"))
    if not feature_ok:
        failures.append("feature_audit.json missing or not passed")

    boundaries = {
        "independently_recomputed_from_raw_arrays": [
            "fold coverage and uniqueness",
            "saved fold labels versus y_train at saved indices",
            "regularization calibration labels versus y_train at seed-42 calibration indices",
            "threshold inner-calibration labels versus y_train at seed-123 calibration indices",
            "confusion matrices, precision, recall, F1, accuracy, balanced accuracy",
            "AUROC and PR-AUC from saved scores",
            "mean F1 and sample SD",
            "paired fold differences",
            "regularization metrics and weight norms",
            "ridge objective identity from saved train_mse and weight_l2",
            "threshold inner metrics from saved inner calibration scores",
            "regularization F1 at optimized and frozen thresholds from saved scores",
        ],
        "targeted_rerun": [
            "threshold_provenance.json was produced by retraining only the inner "
            "210,006-row calibration model; no CV, regularization, final model, "
            "or test prediction was rerun.",
        ],
        "source_inspection_only": [
            "run.py threshold constant",
            "documented feature-header audit",
        ],
        "not_proved": [
            "that the saved CV scores were produced by the included training code "
            "during a specific historical run",
            "full-data final-model weights and test performance",
            "independent generalization of the retrospective CV or regularization "
            "results",
        ],
    }

    result = {
        "passed": not failures,
        "failures": failures,
        "fold_coverage": {"rows": int(y.size), "folds": 5,
                          "rows_per_fold": 65627,
                          "positives_per_fold": 5795,
                          "negatives_per_fold": 59832},
        "recomputed_summary": summary,
        "ridge_objective": {
            "formula": "0.5 * mean((X @ w - y)**2) + ridge_lambda * sum(w**2)",
            "normal_equation_gradient_inf":
                ridge_identity_max_gradient,
        },
        "markdown_checks": markdown_checks,
        "label_provenance": {
            "regularization_calibration": {
                "split_seed": 42,
                "cal_rows": int(regularization_cal.size),
                "labels_match_y_train": regularization_labels_match,
            },
            "threshold_inner_calibration": {
                "split_seed": 123,
                "cal_rows": int(inner_cal.size),
                "labels_match_y_train": threshold_labels_match,
            },
        },
        "threshold_checks": threshold_checks,
        "threshold_provenance": {
            "status": provenance.get("status"),
            "frozen_threshold": provenance.get("frozen_threshold"),
            "regularization_comparison": regression,
        },
        "feature_audit": feature_audit,
        "regularization": {
            "split_seed": regularization["metadata"]["split_seed"],
            "fit_rows": regularization["metadata"]["fit_rows"],
            "cal_rows": regularization["metadata"]["cal_rows"],
            "best_lambda_by_calibration_f1":
                regularization["best_lambda_by_calibration_f1"],
        },
        "verification_boundaries": boundaries,
    }
    with open(os.path.join(output_dir, "verification_report.json"), "w") as handle:
        json.dump(result, handle, indent=2)
    if failures:
        print("FAIL")
        for failure in failures:
            print(" -", failure)
        raise SystemExit(1)
    print("PASS: all saved results independently recomputed.")
    print("fold coverage: 5 x 65,627 rows; 5,795 positives and 59,832 negatives each")
    for model, row in summary.items():
        print("%-24s F1 %.4f +/- %.4f  AUROC %.4f  PR-AUC %.4f" % (
            model, row["f1_mean"], row["f1_sd"], row["auc_mean"], row["pr_mean"]))
    print("Threshold provenance:", provenance.get("status"))
    print("Ridge objective gradient infinity norm: %.2e" % ridge_identity_max_gradient)
    print("Boundary: metrics/labels were recomputed from saved arrays; full "
          "training provenance is not proved by this verifier.")


if __name__ == "__main__":
    main()
