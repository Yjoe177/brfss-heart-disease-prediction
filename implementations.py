"""Mandatory machine learning implementations for Project 1.

This module contains the six functions required by the project handout.
Only NumPy and the Python standard library are used; no external machine
learning or data libraries (Pandas, scikit-learn, PyTorch, ...) are used.

Mathematical conventions follow the course notes:

    MSE loss     L(w) = (1 / (2N)) * || y - X w ||^2
    MSE gradient g(w) = (1 / N) * X^T (X w - y)

    Logistic loss     L(w) = (1 / N) * sum_i [ log(1 + exp(z_i)) - y_i z_i ]
                      with z = X w and labels y in {0, 1}
    Logistic gradient g(w) = (1 / N) * X^T (sigmoid(X w) - y)

Regularized logistic regression adds the term 2 * lambda_ * w to the
logistic gradient. Ridge regression solves the normal equation

    (X^T X + 2 N lambda I) w = X^T y

The loss values returned by the two regularized methods (``ridge_regression``
and ``reg_logistic_regression``) are the *unregularized* data losses, exactly
as required by the handout.

All six functions return ``(w, loss)`` where ``w`` is a one-dimensional NumPy
array of shape ``(D,)`` and ``loss`` is a Python float scalar.
"""

import numpy as np


# ---------------------------------------------------------------------------
# Small shared helpers for the repeated mathematics
# ---------------------------------------------------------------------------
def _mse(y, tx, w):
    """Compute the mean squared error of weights ``w`` on data ``(y, tx)``.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the design matrix (features).
        w: numpy array of shape (D,), the model weights.

    Returns:
        float: the scalar loss (1 / (2N)) * || y - tx @ w ||^2.
    """
    e = y - tx @ w
    return float(np.sum(e * e) / (2.0 * len(e)))


def _mse_gradient(y, tx, w):
    """Compute the gradient of the MSE loss at ``w``.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the design matrix (features).
        w: numpy array of shape (D,), the model weights.

    Returns:
        numpy array of shape (D,): (1 / N) * tx^T (tx @ w - y).
    """
    e = tx @ w - y
    return tx.T @ e / len(e)


def _sigmoid(z):
    """Apply the logistic sigmoid element-wise in a numerically stable way.

    The computation is split on the sign of ``z`` so that ``exp`` is only
    ever applied to a non-positive argument, avoiding overflow for large
    positive or large negative logits.

    Args:
        z: array-like of logits.

    Returns:
        numpy array with the same shape as ``z``, with values in (0, 1).
    """
    z = np.asarray(z, dtype=float)
    out = np.empty_like(z)
    pos = z >= 0
    out[pos] = 1.0 / (1.0 + np.exp(-z[pos]))
    neg = ~pos
    exp_z = np.exp(z[neg])
    out[neg] = exp_z / (1.0 + exp_z)
    return out


def _logistic_loss(y, tx, w):
    """Compute the mean binary logistic loss of ``w`` on ``(y, tx)``.

    Uses ``logaddexp(0, z) = log(1 + exp(z))`` which is stable for large
    positive and negative logits. Labels ``y`` are expected in {0, 1}.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the design matrix (features).
        w: numpy array of shape (D,), the model weights.

    Returns:
        float: (1 / N) * sum_i [ log(1 + exp(z_i)) - y_i z_i ], z = tx @ w.
    """
    z = tx @ w
    return float(np.mean(np.logaddexp(0.0, z) - y * z))


def _logistic_gradient(y, tx, w):
    """Compute the gradient of the logistic loss at ``w``.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the design matrix (features).
        w: numpy array of shape (D,), the model weights.

    Returns:
        numpy array of shape (D,): (1 / N) * tx^T (sigmoid(tx @ w) - y).
    """
    return tx.T @ (_sigmoid(tx @ w) - y) / len(y)


# ---------------------------------------------------------------------------
# Required public functions
# ---------------------------------------------------------------------------
def mean_squared_error_gd(y, tx, initial_w, max_iters, gamma):
    """Linear regression trained with gradient descent.

    Performs exactly ``max_iters`` gradient updates starting from
    ``initial_w`` and returns the final weights together with the MSE loss
    evaluated at those final weights.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the design matrix (features).
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient updates to perform.
        gamma: float, the step size (learning rate).

    Returns:
        tuple: ``(w, loss)`` where ``w`` is a numpy array of shape (D,)
        and ``loss`` is the scalar MSE at ``w``.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        w = w - gamma * _mse_gradient(y, tx, w)
    return w, _mse(y, tx, w)


def mean_squared_error_sgd(y, tx, initial_w, max_iters, gamma):
    """Linear regression trained with stochastic gradient descent.

    Uses mini-batch size 1, matching the course ``batch_iter`` convention:
    at every step a single training sample is drawn uniformly at random and
    one gradient update is applied. Exactly ``max_iters`` updates are done.
    The returned loss is the MSE over the full dataset at the final weights.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the design matrix (features).
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of stochastic gradient updates.
        gamma: float, the step size (learning rate).

    Returns:
        tuple: ``(w, loss)`` where ``w`` is a numpy array of shape (D,)
        and ``loss`` is the scalar MSE at ``w``.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    n = len(y)
    for _ in range(max_iters):
        # Mini-batch of size 1 chosen at random, as in the course batch_iter.
        i = np.random.randint(n, size=1)[0]
        w = w - gamma * _mse_gradient(y[i : i + 1], tx[i : i + 1], w)
    return w, _mse(y, tx, w)


def least_squares(y, tx):
    """Least squares regression using the normal equations.

    Solves ``(tx^T tx) w = tx^T y`` with a linear solver. If the system is
    singular (e.g. collinear features), it falls back to the Moore-Penrose
    pseudo-inverse, which gives the minimum-norm least squares solution.
    ``numpy.linalg.lstsq`` is intentionally not used.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the design matrix (features).

    Returns:
        tuple: ``(w, loss)`` where ``w`` is a numpy array of shape (D,)
        and ``loss`` is the scalar MSE at ``w``.
    """
    xtx = tx.T @ tx
    xty = tx.T @ y
    try:
        w = np.linalg.solve(xtx, xty)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(xtx) @ xty
    return w, _mse(y, tx, w)


def ridge_regression(y, tx, lambda_):
    """Ridge regression using the regularized normal equations.

    Solves ``(tx^T tx + 2 N lambda_ I) w = tx^T y``. The returned loss is the
    ordinary (unregularized) MSE at the solution, as required by the handout.

    Args:
        y: numpy array of shape (N,), the target values.
        tx: numpy array of shape (N, D), the design matrix (features).
        lambda_: float, the non-negative regularization parameter.

    Returns:
        tuple: ``(w, loss)`` where ``w`` is a numpy array of shape (D,)
        and ``loss`` is the scalar unregularized MSE at ``w``.
    """
    n, d = tx.shape
    xtx = tx.T @ tx
    xty = tx.T @ y
    reg = 2.0 * n * lambda_ * np.eye(d)
    try:
        w = np.linalg.solve(xtx + reg, xty)
    except np.linalg.LinAlgError:
        w = np.linalg.pinv(xtx + reg) @ xty
    return w, _mse(y, tx, w)


def logistic_regression(y, tx, initial_w, max_iters, gamma):
    """Logistic regression trained with gradient descent.

    Labels ``y`` are expected in {0, 1}. Performs exactly ``max_iters``
    gradient updates and returns the final weights together with the mean
    logistic loss at those weights.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the design matrix (features).
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient updates to perform.
        gamma: float, the step size (learning rate).

    Returns:
        tuple: ``(w, loss)`` where ``w`` is a numpy array of shape (D,)
        and ``loss`` is the scalar mean logistic loss at ``w``.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        w = w - gamma * _logistic_gradient(y, tx, w)
    return w, _logistic_loss(y, tx, w)


def reg_logistic_regression(y, tx, lambda_, initial_w, max_iters, gamma):
    """Regularized (L2) logistic regression trained with gradient descent.

    Labels ``y`` are expected in {0, 1}. The gradient adds ``2 * lambda_ * w``
    to the plain logistic gradient. The returned loss is the *unregularized*
    mean logistic data loss, as required by the handout.

    Args:
        y: numpy array of shape (N,), labels in {0, 1}.
        tx: numpy array of shape (N, D), the design matrix (features).
        lambda_: float, the non-negative regularization parameter.
        initial_w: numpy array of shape (D,), the initial weights.
        max_iters: int, the number of gradient updates to perform.
        gamma: float, the step size (learning rate).

    Returns:
        tuple: ``(w, loss)`` where ``w`` is a numpy array of shape (D,)
        and ``loss`` is the scalar unregularized mean logistic loss at ``w``.
    """
    w = np.array(initial_w, dtype=float, copy=True)
    for _ in range(max_iters):
        grad = _logistic_gradient(y, tx, w) + 2.0 * lambda_ * w
        w = w - gamma * grad
    return w, _logistic_loss(y, tx, w)
