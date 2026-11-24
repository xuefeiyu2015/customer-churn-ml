"""Integration tests for the training workflow on small synthetic CSVs."""

import numpy as np
import pandas as pd
import pytest
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from customer_churn_ml.persistence import load_metadata, load_model
from customer_churn_ml.preprocessing import FE_NUMERIC, clean_features
from customer_churn_ml.training import (
    RF_PARAMS,
    run_training,
    split_data,
    split_features_target,
)

SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]
N_ROWS = 400


def make_raw(n: int = N_ROWS, seed: int = 0) -> pd.DataFrame:
    """Raw CSV-style rows with a learnable churn signal (contract + tenure)."""
    rng = np.random.default_rng(seed)
    pick = lambda values: rng.choice(values, size=n)  # noqa: E731
    contract = pick(["Month-to-month", "One year", "Two year"])
    tenure = rng.integers(0, 72, size=n)
    monthly = rng.uniform(20, 110, size=n).round(2)
    total = (tenure * monthly).round(2).astype(str)
    total[tenure == 0] = " "  # real-data quirk: blank TotalCharges for new customers
    logit = -1.0 + 1.8 * (contract == "Month-to-month") - 0.04 * tenure
    churn = np.where(rng.random(n) < 1 / (1 + np.exp(-logit)), "Yes", "No")
    return pd.DataFrame({
        "customerID": [f"{i:04d}-SYNT" for i in range(n)],
        "gender": pick(["Female", "Male"]),
        "SeniorCitizen": pick([0, 1]),
        "Partner": pick(["Yes", "No"]),
        "Dependents": pick(["Yes", "No"]),
        "tenure": tenure,
        "PhoneService": pick(["Yes", "No"]),
        "MultipleLines": pick(["Yes", "No", "No phone service"]),
        "InternetService": pick(["DSL", "Fiber optic", "No"]),
        **{s: pick(["Yes", "No", "No internet service"]) for s in SERVICES},
        "Contract": contract,
        "PaperlessBilling": pick(["Yes", "No"]),
        "PaymentMethod": pick(["Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"]),
        "MonthlyCharges": monthly,
        "TotalCharges": total,
        "Churn": churn,
    })


@pytest.fixture(scope="module")
def csv_path(tmp_path_factory):
    path = tmp_path_factory.mktemp("data") / "synthetic.csv"
    make_raw().to_csv(path, index=False)
    return path


@pytest.fixture(scope="module")
def result(csv_path, tmp_path_factory):
    model_path = tmp_path_factory.mktemp("models") / "model.joblib"
    return run_training(csv_path, model_path, include_oof=True)


def test_end_to_end_training_from_raw_csv(result, csv_path):
    pipe = result.pipeline
    assert [name for name, _ in pipe.steps] == ["clean", "prep", "model"]
    assert pipe.named_steps["model"].get_params()["max_depth"] == RF_PARAMS["max_depth"]
    assert {k: pipe.named_steps["model"].get_params()[k] for k in RF_PARAMS} == RF_PARAMS

    names = pipe.named_steps["prep"].get_feature_names_out()
    assert len(names) == 40
    assert not any(n.startswith(("customerid", "churn")) for n in names)

    # pipeline scores raw CSV rows directly (original casing, customerID, blank TotalCharges)
    raw = pd.read_csv(csv_path).drop(columns="Churn")
    proba = pipe.predict_proba(raw)[:, 1]
    assert proba.shape == (N_ROWS,) and ((proba >= 0) & (proba <= 1)).all()

    # synthetic signal is learnable
    assert result.holdout_metrics["roc_auc"] > 0.6


def test_train_test_separation(result, csv_path):
    y = (pd.read_csv(csv_path)["Churn"] == "Yes").astype(int)
    train_idx, test_idx = set(result.train_index), set(result.test_index)

    assert train_idx.isdisjoint(test_idx)
    assert train_idx | test_idx == set(range(N_ROWS))
    assert len(test_idx) == N_ROWS * 0.2

    ref_train, ref_test = train_test_split(
        np.arange(N_ROWS), test_size=0.2, stratify=y, random_state=42
    )
    assert list(result.train_index) == list(ref_train)
    assert list(result.test_index) == list(ref_test)


def test_metrics_match_independent_computation(result, csv_path):
    raw = pd.read_csv(csv_path)
    X_test = raw.loc[result.test_index].drop(columns="Churn")
    y_test = (raw.loc[result.test_index, "Churn"] == "Yes").astype(int).to_numpy()

    model = result.pipeline
    scores = model.predict_proba(X_test)[:, list(model.classes_).index(1)]
    k = round(len(y_test) * 500 / 7043)
    top_k = pd.Series(y_test).iloc[
        pd.Series(scores).sort_values(ascending=False, kind="stable").index[:k]
    ]
    p_at_k = top_k.mean()

    m = result.holdout_metrics
    assert m["k"] == k == 6
    assert m["n"] == len(y_test)
    assert m["average_precision"] == pytest.approx(average_precision_score(y_test, scores))
    assert m["roc_auc"] == pytest.approx(roc_auc_score(y_test, scores))
    assert m["precision_at_k"] == pytest.approx(p_at_k)
    assert m["lift_at_k"] == pytest.approx(p_at_k / y_test.mean())

    assert result.oof_metrics["n"] == len(result.train_index)
    assert result.oof_metrics["k"] == round(len(result.train_index) * 500 / 7043)


def test_save_load_parity_and_metadata(result, csv_path):
    loaded = load_model(result.model_path)
    X = pd.read_csv(csv_path).drop(columns="Churn")
    np.testing.assert_array_equal(loaded.predict_proba(X), result.pipeline.predict_proba(X))

    saved = load_metadata(result.model_path)["metadata"]
    assert saved == result.metadata
    assert saved["holdout_metrics"] == result.holdout_metrics
    assert saved["split"]["n_test"] == len(result.test_index)


def test_training_is_deterministic(result, csv_path, tmp_path):
    again = run_training(csv_path, tmp_path / "model.joblib", include_oof=True)

    assert again.holdout_metrics == result.holdout_metrics
    assert again.oof_metrics == result.oof_metrics
    X = pd.read_csv(csv_path).drop(columns="Churn")
    np.testing.assert_array_equal(
        again.pipeline.predict_proba(X), result.pipeline.predict_proba(X)
    )


@pytest.mark.filterwarnings("ignore:Found unknown categories:UserWarning")
def test_preprocessing_fitted_on_training_rows_only(tmp_path):
    raw = make_raw()
    X, y = split_features_target(raw)
    _, X_test, _, _ = split_data(X, y)

    # Shift the holdout rows only; the split depends only on y, so it is unchanged.
    shifted = raw.copy()
    shifted.loc[X_test.index, "MonthlyCharges"] += 10_000
    shifted.loc[X_test.index, "PaymentMethod"] = "Holdout only"
    path = tmp_path / "shifted.csv"
    shifted.to_csv(path, index=False)

    res = run_training(path, tmp_path / "model.joblib", include_oof=False)
    assert list(res.test_index) == list(X_test.index)

    prep = res.pipeline.named_steps["prep"]
    scaler = prep.named_transformers_["num"]
    train_numeric = clean_features(shifted.loc[res.train_index])[FE_NUMERIC]
    reference = StandardScaler().fit(train_numeric)

    np.testing.assert_array_equal(scaler.mean_, reference.mean_)
    np.testing.assert_array_equal(scaler.var_, reference.var_)
    assert scaler.n_samples_seen_ == len(res.train_index)
    assert scaler.mean_[FE_NUMERIC.index("monthlycharges")] < 200  # holdout rows are > 10,000

    payment_categories = prep.named_transformers_["cat"].categories_[3]
    assert "holdout_only" not in payment_categories
