"""L2 regularization study for plain SGD logistic regression.

Run:
    python experiments/regularization_study.py

The study uses a fixed stratified 80/20 split of all 328,135 labeled samples:
the fitting portion is used for training and the calibration portion for
metrics and threshold selection. It is a separate training-side experiment,
not the five-fold CV; its results are not presented as an independent test.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pipeline as common


def parse_lambdas(values):
    return [float(value) for value in values]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--output-dir", default=os.path.join(HERE, "results"))
    parser.add_argument("--split-seed", type=int, default=42)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--epochs", type=int, default=90)
    parser.add_argument("--schedule", default="inverse")
    parser.add_argument("--decay", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=2027)
    parser.add_argument("--lambdas", nargs="+", default=["0", "1e-5", "1e-4", "1e-3"])
    args = parser.parse_args()

    data_dir = args.data_dir or common.find_data_dir()
    lambdas = parse_lambdas(args.lambdas)
    x, y, header = common.load_train(data_dir)
    rows = np.arange(y.size, dtype=np.int64)
    candidate = common.candidate_columns(header)
    fit_rows, cal_rows = common.stratified_holdout(
        y, rows, args.val_fraction, args.split_seed)
    fit_pos, fit_neg = common.class_counts(y, fit_rows)
    cal_pos, cal_neg = common.class_counts(y, cal_rows)
    print("fit %d (%d pos, %d neg), calibration %d (%d pos, %d neg)" % (
        fit_rows.size, fit_pos, fit_neg, cal_rows.size, cal_pos, cal_neg),
        flush=True)

    prep = common.fit_preprocessor(x, fit_rows, candidate)
    x_fit = common.make_design(x, fit_rows, prep)
    x_cal = common.make_design(x, cal_rows, prep)
    y_fit = (y[fit_rows] == 1).astype(float)
    y_cal = y[cal_rows]
    y_cal_01 = (y_cal == 1).astype(float)
    summary = common.encoded_summary(prep, x_fit.shape[1])
    print("design", x_fit.shape, summary, flush=True)

    results = []
    score_arrays = {"labels": y_cal.astype(np.int8)}
    weight_arrays = {}
    for index, lam in enumerate(lambdas):
        print("lambda %g" % lam, flush=True)
        result = common.train_sgd(
            x_fit, y_fit, lr0=args.lr, epochs=args.epochs,
            schedule=args.schedule, lam=lam, seed=args.seed, decay=args.decay)
        scores = common.sigmoid(x_cal @ result["w"])
        threshold, _ = common.best_threshold(scores, y_cal)
        metrics = common.evaluate_scores(scores, y_cal, threshold)
        calibration_loss = common.logistic_loss(x_cal, y_cal_01, result["w"])
        record = {
            "lambda": lam, "threshold": threshold,
            "metrics": metrics,
            "train_loss": result["data_loss"],
            "penalty": result["penalty"],
            "penalized_train_objective": result["penalized_loss"],
            "calibration_loss": calibration_loss,
            "train_calibration_gap": calibration_loss - result["data_loss"],
            "weight_l2": float(np.linalg.norm(result["w"])),
            "weight_max_abs": float(np.max(np.abs(result["w"]))),
            "grad_inf": result["grad_inf"],
            "history": result["history"],
            "runtime_seconds": result["runtime"],
        }
        results.append(record)
        score_arrays["lambda_%d_scores" % index] = scores
        weight_arrays["lambda_%d_weights" % index] = result["w"]
        print("  F1 %.4f AUC %.4f |w| %.3f train %.5f cal %.5f" % (
            metrics["f1"], metrics["roc_auc"], record["weight_l2"],
            record["train_loss"], calibration_loss), flush=True)

    best = max(results, key=lambda row: (row["metrics"]["f1"], -row["lambda"]))
    output = {
        "metadata": {
            "dataset": "BRFSS 2015 labeled training data",
            "split": "stratified 80/20 training/calibration split of all labeled rows",
            "split_seed": args.split_seed,
            "val_fraction": args.val_fraction,
            "fit_rows": int(fit_rows.size), "cal_rows": int(cal_rows.size),
            "fit_pos": fit_pos, "fit_neg": fit_neg,
            "cal_pos": cal_pos, "cal_neg": cal_neg,
            "candidate_features": int(len(candidate)),
            "preprocessing": summary,
            "optimizer": {"name": "plain single-sample SGD", "lr": args.lr,
                          "schedule": args.schedule, "decay": args.decay,
                          "epochs": args.epochs, "seed": args.seed},
            "lambdas": lambdas,
            "selection_metric": "positive-class F1 on the calibration partition",
            "same_as_earlier_development_split": True,
            "threshold_optimized_on_calibration": True,
            "provenance_note": (
                "This is the same seed-42 80/20 partition used during earlier "
                "development. F1 values use thresholds optimized on this same "
                "calibration portion; they describe model selection and "
                "regularization behavior, not independent generalization."),
        },
        "results": results,
        "best_lambda_by_calibration_f1": best["lambda"],
    }

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "regularization_results.json"), "w") as handle:
        json.dump(output, handle, indent=2)
    np.savez(os.path.join(args.output_dir, "regularization_scores.npz"),
             **score_arrays)
    np.savez(os.path.join(args.output_dir, "regularization_weights.npz"),
             lambdas=np.asarray(lambdas), **weight_arrays)

    lines = ["# L2 regularization study", "",
             "Training-side stratified 80/20 split of all 328,135 labeled samples "
             "(split seed %d)." % args.split_seed, "",
             "This is the same seed-42 80/20 partition used during earlier "
             "development. F1 values use thresholds optimized on this same "
             "calibration portion; they describe model selection and "
             "regularization behavior, not independent generalization.", "",
             "| lambda | F1 | Precision | Recall | AUROC | PR-AUC | Train loss | Calibration loss | |w| |",
             "| ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |"]
    for row in results:
        m = row["metrics"]
        lines.append("| %g | %.4f | %.4f | %.4f | %.4f | %.4f | %.5f | %.5f | %.3f |" % (
            row["lambda"], m["f1"], m["precision"], m["recall"], m["roc_auc"],
            m["pr_auc"], row["train_loss"], row["calibration_loss"],
            row["weight_l2"]))
    lines += ["", "Best calibration F1: lambda = %g." % best["lambda"]]
    with open(os.path.join(args.output_dir, "regularization_summary.md"), "w") as handle:
        handle.write("\n".join(lines))
    print("Saved regularization outputs to", args.output_dir)
    print("Best calibration F1 lambda:", best["lambda"])


if __name__ == "__main__":
    main()
