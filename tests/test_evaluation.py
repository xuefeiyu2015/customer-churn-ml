import numpy as np
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score

from customer_churn_ml.evaluation import (
    compute_k,
    compute_metrics,
    lift_at_k,
    positive_class_proba,
    precision_at_k,
)

Y_TRUE = np.array([1, 0, 0, 1, 0, 1, 0, 0])
Y_SCORE = np.array([0.9, 0.8, 0.1, 0.2, 0.7, 0.6, 0.3, 0.05])
# Ranked by score: 0.9(1) 0.8(0) 0.7(0) 0.6(1) 0.3(0) 0.2(1) 0.1(0) 0.05(0)


@pytest.mark.parametrize("k, expected", [(1, 1.0), (2, 0.5), (4, 0.5), (8, 3 / 8)])
def test_precision_at_k_hand_computed(k, expected):
    assert precision_at_k(Y_TRUE, Y_SCORE, k) == pytest.approx(expected)


def test_precision_at_k_rejects_2d_scores():
    """Regression: (n, 1) scores made the notebook's holdout P@100 = 0.0."""
    with pytest.raises(ValueError, match="1D"):
        precision_at_k(Y_TRUE, Y_SCORE.reshape(-1, 1), 2)


@pytest.mark.parametrize("k", [0, 9, 2.0, True])
def test_precision_at_k_rejects_invalid_k(k):
    with pytest.raises(ValueError):
        precision_at_k(Y_TRUE, Y_SCORE, k)


def test_precision_at_k_ties_are_deterministic():
    y_true = np.array([0, 1, 0, 1])
    y_score = np.array([0.5, 0.5, 0.5, 0.5])
    # stable sort keeps original order -> top-2 are indices 0, 1
    assert precision_at_k(y_true, y_score, 2) == 0.5


def test_lift_at_k():
    # P@2 = 0.5, prevalence = 3/8
    assert lift_at_k(Y_TRUE, Y_SCORE, 2) == pytest.approx(0.5 / (3 / 8))


def test_compute_k_matches_notebook_scaling():
    assert compute_k(5634) == 400
    assert compute_k(1409) == 100
    assert compute_k(7043) == 500


class _StubModel:
    classes_ = np.array([1, 0])  # positive class deliberately in column 0

    def predict_proba(self, X):
        p = np.asarray(X, dtype=float)
        return np.column_stack([p, 1 - p])


def test_positive_class_proba_uses_classes_and_returns_1d():
    proba = positive_class_proba(_StubModel(), [0.2, 0.7])
    assert proba.shape == (2,)
    np.testing.assert_allclose(proba, [0.2, 0.7])


def test_compute_metrics_matches_sklearn():
    m = compute_metrics(Y_TRUE, Y_SCORE, k=2)
    assert m["average_precision"] == pytest.approx(average_precision_score(Y_TRUE, Y_SCORE))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(Y_TRUE, Y_SCORE))
    assert m["precision_at_k"] == pytest.approx(0.5)
    assert m["k"] == 2
    assert m["prevalence"] == pytest.approx(3 / 8)


def test_compute_metrics_rejects_single_class():
    with pytest.raises(ValueError, match="single class"):
        compute_metrics(np.zeros(4, dtype=int), np.array([0.1, 0.2, 0.3, 0.4]), k=2)
