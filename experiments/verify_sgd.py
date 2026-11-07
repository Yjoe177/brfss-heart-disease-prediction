"""Verify the hand-written logistic SGD implementation.

Run:
    python experiments/verify_sgd.py

The checks cover analytical versus numerical gradients, the L2 penalty, a
manual single-example update, the epoch-indexed learning-rate schedule,
deterministic shuffling, finite training behaviour, and the actual one-sample
training update.
"""

import argparse
import json
import os
import sys

import numpy as np

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.dirname(HERE))
import pipeline as common


def objective(x, y, w, lam):
    return common.logistic_loss(x, y, w) + lam * float(np.sum(w * w))


def numerical_gradient(x, y, w, lam, epsilon=1e-6):
    gradient = np.zeros_like(w)
    for j in range(w.size):
        plus = w.copy()
        minus = w.copy()
        plus[j] += epsilon
        minus[j] -= epsilon
        gradient[j] = (objective(x, y, plus, lam) -
                       objective(x, y, minus, lam)) / (2.0 * epsilon)
    return gradient


def run_checks():
    rng = np.random.default_rng(11)
    n, d = 120, 6
    x = rng.normal(size=(n, d))
    x[:, 0] = 1.0
    y = (rng.random(n) > 0.45).astype(float)
    w = rng.normal(size=d)
    checks = {}

    for lam in (0.0, 1e-3):
        analytic = common.logistic_grad(x, y, w, lam)
        numeric = numerical_gradient(x, y, w, lam)
        checks["gradient_lambda_%g" % lam] = bool(
            np.allclose(analytic, numeric, atol=1e-6, rtol=1e-5))

    individual = np.array([
        (common.sigmoid_scalar(float(x[i] @ w)) - y[i]) * x[i]
        for i in range(n)])
    checks["average_individual_gradient"] = bool(np.allclose(
        individual.mean(axis=0), common.logistic_grad(x, y, w, 0.0),
        atol=1e-12))

    i = 4
    eta = 0.03
    lam = 1e-5
    probability = common.sigmoid_scalar(float(x[i] @ w))
    manual = w - eta * ((probability - y[i]) * x[i] + 2.0 * lam * w)
    updated = w.copy()
    updated -= eta * ((common.sigmoid_scalar(float(x[i] @ updated)) - y[i]) *
                      x[i] + 2.0 * lam * updated)
    checks["single_sample_update_l2"] = bool(np.allclose(manual, updated))

    expected = [0.01, 0.01 / 1.1, 0.01 / (1.0 + 0.1 * 89)]
    observed = [common.learning_rate(t, 0.01, 0.1) for t in (1, 2, 90)]
    checks["schedule_values"] = bool(np.allclose(expected, observed))

    first = np.random.default_rng(7).permutation(100)
    second = np.random.default_rng(7).permutation(100)
    third = np.random.default_rng(8).permutation(100)
    checks["shuffle_reproducible"] = bool(np.array_equal(first, second))
    checks["shuffle_seed_changes_order"] = bool(not np.array_equal(first, third))

    xs = rng.normal(size=(600, 6))
    xs[:, 0] = 1.0
    true_w = np.array([-0.5, 1.2, -0.8, 0.5, 0.0, -0.4])
    ys = (common.sigmoid(xs @ true_w) > 0.5).astype(float)
    result = common.train_sgd(xs, ys, lr0=0.05, epochs=30,
                              schedule="inverse", lam=0.0, seed=11)
    losses = [row["data_loss"] for row in result["history"]]
    checks["synthetic_loss_decreases"] = bool(losses[-1] < losses[0])
    checks["all_finite"] = bool(np.all(np.isfinite(result["w"])) and
                                np.isfinite(result["data_loss"]) and
                                np.isfinite(result["penalized_loss"]))
    checks["objective_consistent"] = bool(abs(
        result["data_loss"] + result["penalty"] -
        result["penalized_loss"]) < 1e-12)

    one_x = np.array([[1.0, 2.0]])
    one_y = np.array([1.0])
    one_eta = 0.03
    one_result = common.train_sgd(
        one_x, one_y, lr0=one_eta, epochs=1, schedule="inverse",
        lam=1e-5, seed=11, eval_every=1)
    expected_one_step = -one_eta * (0.5 - one_y[0]) * one_x[0]
    checks["single_sample_training_step"] = bool(np.allclose(
        one_result["w"], expected_one_step, atol=1e-12))

    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", default=os.path.join(
        HERE, "results", "verify_sgd.json"))
    args = parser.parse_args()
    checks = run_checks()
    for name, passed in checks.items():
        print("%s %s" % ("PASS" if passed else "FAIL", name))
    result = {"passed": bool(all(checks.values())), "checks": checks,
              "configuration": {"learning_rate": 0.01, "decay": 0.1,
                                "epochs": 90, "lambda": 1e-5, "seed": 2027}}
    os.makedirs(os.path.dirname(args.output), exist_ok=True)
    with open(args.output, "w") as handle:
        json.dump(result, handle, indent=2)
    print("Saved", args.output)
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
