"""Stratified five-fold cross-validation on all 328,135 labeled examples.

Run:
    python experiments/fivefold_cv.py

Every fold holds out 65,627 examples (5,795 positives, 59,832 negatives).
Preprocessing and the decision threshold are fitted on training-side data only.
The saved raw held-out scores allow the metrics to be recomputed independently.
"""

import argparse
import hashlib
import json
import os
import sys
import time

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pipeline as common

MODELS = ["majority", "ridge", "logistic_numeric", "logistic_encoded",
          "reg_logistic_encoded"]
EXPECTED_FOLD = {"eval_rows": 65627, "eval_pos": 5795, "eval_neg": 59832,
                 "train_rows": 262508, "train_pos": 23180, "train_neg": 239328}


def majority_metrics(y):
    positive = int(np.sum(y == 1))
    negative = int(np.sum(y == -1))
    return {
        "threshold": None, "TP": 0, "FP": 0, "TN": negative, "FN": positive,
        "accuracy": float(negative) / float(y.size), "precision": 0.0,
        "recall": 0.0, "f1": 0.0, "specificity": 1.0,
        "balanced_accuracy": 0.5,
        "roc_auc": 0.5, "pr_auc": float(positive) / float(y.size),
    }


def threshold_from_inner(kind, x_fit, x_cal, y_fit, y_cal, args, lam):
    if kind == "ridge":
        weights = common.fit_linear(x_fit, y_fit, args.ridge_lambda)
        scores = x_cal @ weights
        threshold, f1 = common.best_threshold(scores, y_cal)
        return threshold, f1
    result = common.train_sgd(
        x_fit, y_fit, lr0=args.lr, epochs=args.epochs,
        schedule=args.schedule, lam=lam, seed=args.sgd_seed, decay=args.decay)
    scores = common.sigmoid(x_cal @ result["w"])
    threshold, f1 = common.best_threshold(scores, y_cal)
    return threshold, f1


def train_outer(kind, x_train, x_eval, y_train, args, lam):
    start = time.time()
    if kind == "ridge":
        weights = common.fit_linear(x_train, y_train, args.ridge_lambda)
        scores = x_eval @ weights
        train_scores = x_train @ weights
        train_mse = float(np.mean((train_scores - y_train) ** 2))
        half_train_mse = 0.5 * train_mse
        weight_l2 = float(np.linalg.norm(weights))
        return {
            "weights": weights, "scores": scores,
            "train_mse": train_mse,
            "half_train_mse": half_train_mse,
            "penalized_objective": half_train_mse +
            args.ridge_lambda * weight_l2 ** 2,
            "weight_l2": weight_l2, "grad_inf": None, "history": [],
            "runtime_seconds": time.time() - start,
        }
    result = common.train_sgd(
        x_train, y_train, lr0=args.lr, epochs=args.epochs,
        schedule=args.schedule, lam=lam, seed=args.sgd_seed, decay=args.decay)
    return {
        "weights": result["w"],
        "scores": common.sigmoid(x_eval @ result["w"]),
        "train_loss": result["data_loss"],
        "penalized_train_objective": result["penalized_loss"],
        "weight_l2": float(np.linalg.norm(result["w"])),
        "grad_inf": result["grad_inf"],
        "history": result["history"],
        "runtime_seconds": time.time() - start,
    }


def file_sha256(path):
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def fold_config_hash(args, data_dir):
    payload = {
        "fold_seed": args.fold_seed, "sgd_seed": args.sgd_seed,
        "n_folds": args.n_folds, "lr": args.lr, "epochs": args.epochs,
        "schedule": args.schedule, "decay": args.decay,
        "lambda_l2": args.lambda_l2, "ridge_lambda": args.ridge_lambda,
        "models": MODELS,
        "source_sha256": {
            "fivefold_cv.py": file_sha256(__file__),
            "pipeline.py": file_sha256(common.__file__),
        },
        "data_sha256": {
            name: file_sha256(os.path.join(data_dir, name))
            for name in ("x_train.csv", "y_train.csv")
        },
    }
    return hashlib.md5(json.dumps(payload, sort_keys=True).encode()).hexdigest()


def evaluate_fold(fold_index, folds, x, y, candidate, args, config_hash):
    outer_eval = folds[fold_index]
    outer_train = np.setdiff1d(np.arange(y.size, dtype=np.int64), outer_eval)
    eval_pos, eval_neg = common.class_counts(y, outer_eval)
    train_pos, train_neg = common.class_counts(y, outer_train)
    observed = {"eval_rows": int(outer_eval.size), "eval_pos": eval_pos,
                "eval_neg": eval_neg, "train_rows": int(outer_train.size),
                "train_pos": train_pos, "train_neg": train_neg}
    for key, expected in EXPECTED_FOLD.items():
        if observed[key] != expected:
            raise RuntimeError("Fold %d: %s=%d, expected %d" % (
                fold_index, key, observed[key], expected))
    if np.intersect1d(outer_train, outer_eval).size:
        raise RuntimeError("Training and evaluation rows overlap.")

    inner_seed = args.fold_seed * 10 + fold_index + 100000
    inner_fit, inner_cal = common.stratified_holdout(
        y, outer_train, 0.2, inner_seed)
    print("fold %d: eval %d, train %d, inner fit %d, inner cal %d" % (
        fold_index, outer_eval.size, outer_train.size, inner_fit.size,
        inner_cal.size), flush=True)

    # Inner preprocessing and thresholds.
    encoded_inner = common.fit_preprocessor(x, inner_fit, candidate)
    encoded_fit = common.make_design(x, inner_fit, encoded_inner)
    encoded_cal = common.make_design(x, inner_cal, encoded_inner)
    numeric_inner = common.fit_numeric_preprocessor(x, inner_fit, candidate)
    numeric_fit = common.make_numeric_design(x, inner_fit, numeric_inner)
    numeric_cal = common.make_numeric_design(x, inner_cal, numeric_inner)
    y_inner_fit = (y[inner_fit] == 1).astype(float)
    y_inner_cal = y[inner_cal]

    thresholds = {}
    inner_f1 = {}
    thresholds["ridge"], inner_f1["ridge"] = threshold_from_inner(
        "ridge", encoded_fit, encoded_cal, y_inner_fit, y_inner_cal,
        args, 0.0)
    for model, kind, fit, cal, lam in (
            ("logistic_numeric", "logistic", numeric_fit, numeric_cal, 0.0),
            ("logistic_encoded", "logistic", encoded_fit, encoded_cal, 0.0),
            ("reg_logistic_encoded", "logistic", encoded_fit, encoded_cal,
             args.lambda_l2)):
        thresholds[model], inner_f1[model] = threshold_from_inner(
            kind, fit, cal, y_inner_fit, y_inner_cal, args, lam)
    inner_f1["majority"] = 0.0

    # Outer preprocessing; fitted only on the outer training portion.
    encoded_outer = common.fit_preprocessor(x, outer_train, candidate)
    encoded_train = common.make_design(x, outer_train, encoded_outer)
    encoded_eval = common.make_design(x, outer_eval, encoded_outer)
    numeric_outer = common.fit_numeric_preprocessor(x, outer_train, candidate)
    numeric_train = common.make_numeric_design(x, outer_train, numeric_outer)
    numeric_eval = common.make_numeric_design(x, outer_eval, numeric_outer)
    y_train = (y[outer_train] == 1).astype(float)
    y_eval = y[outer_eval]

    encoded_info = common.encoded_summary(encoded_outer, encoded_train.shape[1])
    numeric_info = common.numeric_summary(numeric_outer, numeric_train.shape[1])
    records = {}
    scores_payload = {"labels": y_eval.astype(np.int8)}

    # Majority baseline.
    prediction = -np.ones(outer_eval.size, dtype=np.int64)
    records["majority"] = {
        "model": "majority", "threshold": None,
        "inner_calibration_f1": 0.0,
        "metrics": majority_metrics(y_eval),
        "preprocessing": {"design_dimension": 1},
        "optimizer": {"name": "always negative"},
        "regularization": None,
        "train_loss": None, "penalized_train_objective": None,
        "weight_l2": None, "grad_inf": None, "history": [],
        "runtime_seconds": 0.0,
    }
    scores_payload["majority_prediction"] = prediction.astype(np.int8)

    # Ridge.
    ridge = train_outer("ridge", encoded_train, encoded_eval, y_train,
                        args, 0.0)
    records["ridge"] = {
        "model": "ridge", "threshold": thresholds["ridge"],
        "inner_calibration_f1": inner_f1["ridge"],
        "metrics": common.evaluate_scores(
            ridge["scores"], y_eval, thresholds["ridge"]),
        "preprocessing": encoded_info,
        "optimizer": {"name": "normal equations", "ridge_lambda": args.ridge_lambda},
        "regularization": {"lambda": args.ridge_lambda, "penalty": "all weights"},
        "train_mse": ridge["train_mse"],
        "half_train_mse": ridge["half_train_mse"],
        "ridge_objective_definition":
            "0.5 * mean((X @ w - y)**2) + ridge_lambda * sum(w**2)",
        "penalized_train_objective": ridge["penalized_objective"],
        "weight_l2": ridge["weight_l2"], "grad_inf": None, "history": [],
        "runtime_seconds": ridge["runtime_seconds"],
    }
    scores_payload["ridge_scores"] = ridge["scores"]

    # SGD logistic models.
    for model, x_train, x_eval, info, lam in (
            ("logistic_numeric", numeric_train, numeric_eval, numeric_info, 0.0),
            ("logistic_encoded", encoded_train, encoded_eval, encoded_info, 0.0),
            ("reg_logistic_encoded", encoded_train, encoded_eval, encoded_info,
             args.lambda_l2)):
        fit = train_outer("logistic", x_train, x_eval, y_train,
                          args, lam)
        records[model] = {
            "model": model, "threshold": thresholds[model],
            "inner_calibration_f1": inner_f1[model],
            "metrics": common.evaluate_scores(
                fit["scores"], y_eval, thresholds[model]),
            "preprocessing": info,
            "optimizer": {"name": "plain single-sample SGD", "lr": args.lr,
                          "schedule": args.schedule, "decay": args.decay,
                          "epochs": args.epochs, "seed": args.sgd_seed},
            "regularization": {"lambda": lam, "penalty": "all weights"},
            "train_loss": fit["train_loss"],
            "penalized_train_objective": fit["penalized_train_objective"],
            "weight_l2": fit["weight_l2"], "grad_inf": fit["grad_inf"],
            "history": fit["history"],
            "runtime_seconds": fit["runtime_seconds"],
        }
        scores_payload[model + "_scores"] = fit["scores"]

    result = {
        "fold": fold_index, "config_hash": config_hash,
        "inner_split_seed": inner_seed,
        "counts": observed,
        "inner_counts": {
            "fit_rows": int(inner_fit.size),
            "cal_rows": int(inner_cal.size),
            "fit_pos": int(np.sum(y[inner_fit] == 1)),
            "fit_neg": int(np.sum(y[inner_fit] == -1)),
            "cal_pos": int(np.sum(y[inner_cal] == 1)),
            "cal_neg": int(np.sum(y[inner_cal] == -1)),
        },
        "thresholds": thresholds,
        "models": records,
    }
    return result, scores_payload


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--fold-seed", type=int, default=2028)
    parser.add_argument("--sgd-seed", type=int, default=2027)
    parser.add_argument("--n-folds", type=int, default=5)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--schedule", default="inverse")
    parser.add_argument("--decay", type=float, default=0.1)
    parser.add_argument("--lambda-l2", type=float, default=1e-5)
    parser.add_argument("--ridge-lambda", type=float, default=1e-4)
    parser.add_argument("--resume", action="store_true")
    args = parser.parse_args()

    data_dir = args.data_dir or common.find_data_dir()
    os.makedirs(args.output_dir, exist_ok=True)
    work_dir = os.path.join(args.output_dir, ".work")
    os.makedirs(work_dir, exist_ok=True)
    config_hash = fold_config_hash(args, data_dir)

    x, y, header = common.load_train(data_dir)
    rows = np.arange(y.size, dtype=np.int64)
    candidate = common.candidate_columns(header)
    folds = common.stratified_folds(y, rows, args.n_folds, args.fold_seed)
    np.savez(os.path.join(args.output_dir, "folds.npz"),
             seed=args.fold_seed, n_folds=args.n_folds,
             **{"fold%d" % k: folds[k] for k in range(args.n_folds)})
    print("dataset %d, positives %d, negatives %d" % (
        y.size, int(np.sum(y == 1)), int(np.sum(y == -1))), flush=True)
    for k, fold in enumerate(folds):
        pos, neg = common.class_counts(y, fold)
        print("fold %d rows %d pos %d neg %d" % (k, fold.size, pos, neg),
              flush=True)

    fold_records = []
    score_payloads = []
    for k in range(args.n_folds):
        record_path = os.path.join(work_dir, "fold%d.json" % k)
        scores_path = os.path.join(work_dir, "scores_fold%d.npz" % k)
        if args.resume and os.path.exists(record_path) and os.path.exists(scores_path):
            with open(record_path) as handle:
                saved = json.load(handle)
            if saved.get("config_hash") == config_hash:
                fold_records.append(saved)
                score_payloads.append(dict(np.load(scores_path)))
                print("skip fold %d (checkpoint)" % k, flush=True)
                continue
        record, payload = evaluate_fold(
            k, folds, x, y, candidate, args, config_hash)
        with open(record_path, "w") as handle:
            json.dump(record, handle, indent=2)
        np.savez(scores_path, **payload)
        fold_records.append(record)
        score_payloads.append(payload)
        print("fold %d complete" % k, flush=True)

    summary = {}
    for model in MODELS:
        rows_for_model = [record["models"][model] for record in fold_records]
        f1 = np.array([row["metrics"]["f1"] for row in rows_for_model])
        auc = np.array([row["metrics"]["roc_auc"] for row in rows_for_model])
        pr = np.array([row["metrics"]["pr_auc"] for row in rows_for_model])
        summary[model] = {
            "f1_mean": float(f1.mean()), "f1_sd": float(f1.std(ddof=1)),
            "auc_mean": float(auc.mean()), "auc_sd": float(auc.std(ddof=1)),
            "pr_mean": float(pr.mean()), "pr_sd": float(pr.std(ddof=1)),
            "f1_values": f1.tolist(), "auc_values": auc.tolist(),
            "pr_values": pr.tolist(),
        }

    def paired(model_a, model_b):
        values = np.array([
            record["models"][model_a]["metrics"]["f1"] -
            record["models"][model_b]["metrics"]["f1"]
            for record in fold_records])
        return {
            "model_a": model_a, "model_b": model_b,
            "direction": "F1(model_a) - F1(model_b)",
            "mean": float(values.mean()),
            "sample_sd": float(values.std(ddof=1)),
            "values": values.tolist(),
        }

    output = {
        "metadata": {
            "dataset": "all 328,135 labeled BRFSS training examples",
            "n_rows": int(y.size),
            "positives": int(np.sum(y == 1)),
            "negatives": int(np.sum(y == -1)),
            "n_folds": args.n_folds, "fold_seed": args.fold_seed,
            "protocol": "stratified five-fold CV; nested inner 80/20 calibration split per fold",
            "evaluation_note": "retrospective evaluation of a frozen procedure; folds are correlated and the labels participated in earlier development",
            "models": MODELS,
            "optimizer": {"name": "plain single-sample SGD", "lr": args.lr,
                          "schedule": args.schedule, "decay": args.decay,
                          "epochs": args.epochs, "seed": args.sgd_seed},
            "lambda_l2": args.lambda_l2,
            "ridge_lambda": args.ridge_lambda,
        },
        "folds": fold_records,
        "summary": summary,
        "paired_differences": {
            "reg_logistic_minus_unregularized": paired(
                "reg_logistic_encoded", "logistic_encoded"),
            "encoded_minus_numeric": paired(
                "logistic_encoded", "logistic_numeric"),
            "unregularized_minus_ridge": paired(
                "logistic_encoded", "ridge"),
        },
    }
    with open(os.path.join(args.output_dir, "fivefold_results.json"), "w") as handle:
        json.dump(output, handle, indent=2)

    combined = {}
    for k, payload in enumerate(score_payloads):
        for key, value in payload.items():
            combined["fold%d_%s" % (k, key)] = value
    np.savez(os.path.join(args.output_dir, "fivefold_scores.npz"), **combined)

    lines = ["# Full-data stratified five-fold CV", "",
             "All 328,135 labeled examples; held-out folds of 65,627 rows "
             "(5,795 positives, 59,832 negatives).", "",
             "Retrospective evaluation of a frozen procedure: all labeled rows "
             "informed earlier development and the folds are correlated.", "",
             "| Model | F1 mean +/- sample SD | AUROC | PR-AUC |",
             "| --- | ---: | ---: | ---: |"]
    for model in MODELS:
        row = summary[model]
        lines.append("| %s | %.4f +/- %.4f | %.4f | %.4f |" % (
            model, row["f1_mean"], row["f1_sd"], row["auc_mean"], row["pr_mean"]))
    lines += ["", "Paired differences (sample SD over five folds):", ""]
    for name, value in output["paired_differences"].items():
        lines.append("- F1(%s) - F1(%s): %+.4f +/- %.4f" % (
            value["model_a"], value["model_b"], value["mean"],
            value["sample_sd"]))
    with open(os.path.join(args.output_dir, "fivefold_summary.md"), "w") as handle:
        handle.write("\n".join(lines))
    print("Saved five-fold outputs to", args.output_dir, flush=True)


if __name__ == "__main__":
    main()
