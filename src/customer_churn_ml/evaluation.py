"""Ranking metrics for churn-risk scores.

All functions are pure: they take labels/scores and return numbers, with no
plotting, printing, or file I/O.

K scaling: the business scenario assumes capacity to contact 500 of 7043
customers (~7.1%). Evaluation sets are smaller, so K is scaled to the size of
the evaluated population (OOF on 5634 rows -> K=400, holdout on 1409 -> K=100)
rather than using K=500 directly.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import ArrayLike
from sklearn.metrics import average_precision_score, roc_auc_score

CONTACT_CAPACITY = 500
REFERENCE_POPULATION = 7043
POSITIVE_LABEL = 1


def _validate_inputs(y_true: ArrayLike, y_score: ArrayLike) -> tuple[np.ndarray, np.ndarray]:
    """Return 1D numpy arrays; reject shape mismatches and non-binary labels.

    Rejecting 2D scores (e.g. shape (n, 1)) guards against the notebook bug in
    which a boolean column mask produced a 2D probability array, making
    ``np.argsort`` sort along the wrong axis.
    """
    y_true_arr = np.asarray(y_true)
    y_score_arr = np.asarray(y_score, dtype=float)

    if y_true_arr.ndim != 1:
        raise ValueError(f"y_true must be 1D, got shape {y_true_arr.shape}")
    if y_score_arr.ndim != 1:
        raise ValueError(f"y_score must be 1D, got shape {y_score_arr.shape}")
    if len(y_true_arr) != len(y_score_arr):
        raise ValueError(
            f"y_true and y_score lengths differ: {len(y_true_arr)} vs {len(y_score_arr)}"
        )
    if len(y_true_arr) == 0:
        raise ValueError("y_true is empty")
    if not np.isin(y_true_arr, [0, 1]).all():
        raise ValueError("y_true must contain only 0/1 labels")
    if not np.isfinite(y_score_arr).all():
        raise ValueError("y_score contains non-finite values")

    return y_true_arr.astype(int), y_score_arr


def compute_k(
    n_eval: int,
    capacity: int = CONTACT_CAPACITY,
    population: int = REFERENCE_POPULATION,
) -> int:
    """Scale the contact capacity to an evaluation set of size ``n_eval``."""
    if n_eval < 1:
        raise ValueError("n_eval must be >= 1")
    if not 1 <= capacity <= population:
        raise ValueError("capacity must be between 1 and population")
    return max(1, round(n_eval * capacity / population))


def precision_at_k(y_true: ArrayLike, y_score: ArrayLike, k: int) -> float:
    """Fraction of actual churners among the ``k`` highest-scored customers.

    Ties are broken by original order (stable sort on negated scores), so the
    result is deterministic.
    """
    y_true_arr, y_score_arr = _validate_inputs(y_true, y_score)

    if not isinstance(k, (int, np.integer)) or isinstance(k, (bool, np.bool_)):
        raise ValueError("k must be an integer")
    if not 1 <= k <= len(y_true_arr):
        raise ValueError("k must be between 1 and the number of samples")

    order = np.argsort(-y_score_arr, kind="stable")
    return float(y_true_arr[order[:k]].mean())


def lift_at_k(y_true: ArrayLike, y_score: ArrayLike, k: int) -> float:
    """Precision@K divided by the base churn rate (random-selection precision)."""
    y_true_arr, _ = _validate_inputs(y_true, y_score)
    prevalence = y_true_arr.mean()
    if prevalence == 0:
        raise ValueError("lift is undefined when y_true has no positives")
    return precision_at_k(y_true, y_score, k) / prevalence


def positive_class_proba(model, X) -> np.ndarray:
    """Return a 1D vector of P(churn=1) from a fitted classifier or Pipeline."""
    classes = np.asarray(model.classes_)
    matches = np.flatnonzero(classes == POSITIVE_LABEL)
    if len(matches) != 1:
        raise ValueError(f"positive label {POSITIVE_LABEL} not found in classes_ {classes}")
    return np.asarray(model.predict_proba(X))[:, int(matches[0])]


def compute_metrics(y_true: ArrayLike, y_score: ArrayLike, k: int) -> dict[str, float | int]:
    """AP, ROC AUC, Precision@K and Lift@K for one evaluation set."""
    y_true_arr, y_score_arr = _validate_inputs(y_true, y_score)
    if len(np.unique(y_true_arr)) < 2:
        raise ValueError("y_true contains a single class; ROC AUC is undefined")

    return {
        "average_precision": float(average_precision_score(y_true_arr, y_score_arr)),
        "roc_auc": float(roc_auc_score(y_true_arr, y_score_arr)),
        "precision_at_k": precision_at_k(y_true_arr, y_score_arr, k),
        "lift_at_k": lift_at_k(y_true_arr, y_score_arr, k),
        "k": int(k),
        "prevalence": float(y_true_arr.mean()),
        "n": int(len(y_true_arr)),
    }
