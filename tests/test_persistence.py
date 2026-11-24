import numpy as np
import pandas as pd
import pytest
import sklearn
from sklearn.exceptions import NotFittedError
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from customer_churn_ml.persistence import load_metadata, load_model, save_model
from customer_churn_ml.preprocessing import build_preprocessor, clean_features

SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]


def make_raw(n: int = 60, seed: int = 0) -> pd.DataFrame:
    """Synthetic raw CSV-style rows (original column names and casing)."""
    rng = np.random.default_rng(seed)
    pick = lambda values: rng.choice(values, size=n)  # noqa: E731
    tenure = rng.integers(1, 72, size=n)
    monthly = rng.uniform(20, 110, size=n).round(2)
    return pd.DataFrame({
        "customerID": [f"{i:04d}-TEST" for i in range(n)],
        "gender": pick(["Female", "Male"]),
        "SeniorCitizen": pick([0, 1]),
        "Partner": pick(["Yes", "No"]),
        "Dependents": pick(["Yes", "No"]),
        "tenure": tenure,
        "PhoneService": pick(["Yes", "No"]),
        "MultipleLines": pick(["Yes", "No", "No phone service"]),
        "InternetService": pick(["DSL", "Fiber optic", "No"]),
        **{s: pick(["Yes", "No", "No internet service"]) for s in SERVICES},
        "Contract": pick(["Month-to-month", "One year", "Two year"]),
        "PaperlessBilling": pick(["Yes", "No"]),
        "PaymentMethod": pick(["Electronic check", "Mailed check",
                               "Bank transfer (automatic)", "Credit card (automatic)"]),
        "MonthlyCharges": monthly,
        "TotalCharges": (tenure * monthly).round(2).astype(str),
    })


def make_pipeline() -> Pipeline:
    return Pipeline([
        ("clean", FunctionTransformer(clean_features)),
        ("prep", build_preprocessor()),
        ("model", LogisticRegression(max_iter=1000)),
    ])


@pytest.fixture(scope="module")
def raw() -> pd.DataFrame:
    return make_raw()


@pytest.fixture(scope="module")
def fitted(raw) -> Pipeline:
    y = np.random.default_rng(1).integers(0, 2, size=len(raw))
    return make_pipeline().fit(raw, y)


def test_save_load_roundtrip_identical_predictions(tmp_path, raw, fitted):
    path, _ = save_model(fitted, tmp_path / "model.joblib", {"seed": 42})
    loaded = load_model(path)

    assert [name for name, _ in loaded.steps] == ["clean", "prep", "model"]
    # loaded pipeline accepts raw CSV-style rows directly
    np.testing.assert_array_equal(loaded.predict_proba(raw), fitted.predict_proba(raw))


def test_metadata_saved(tmp_path, fitted):
    metadata = {"model": "RandomForestClassifier", "random_state": 42,
                "metrics": {"average_precision": 0.6588, "k": 100}}
    model_path, meta_path = save_model(fitted, tmp_path / "model.joblib", metadata)

    assert meta_path == tmp_path / "model.json"
    saved = load_metadata(model_path)
    assert saved["metadata"] == metadata
    assert saved["environment"]["scikit-learn"] == sklearn.__version__


def test_missing_model_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError, match="Model file not found"):
        load_model(tmp_path / "does_not_exist.joblib")


def test_parent_directories_created(tmp_path, fitted):
    path, meta = save_model(fitted, tmp_path / "models" / "v1" / "model.joblib")
    assert path.is_file() and meta.is_file()


def test_overwrite_prevented_unless_allowed(tmp_path, raw, fitted):
    path, _ = save_model(fitted, tmp_path / "model.joblib", {"version": 1})
    original_bytes = path.read_bytes()

    with pytest.raises(FileExistsError, match="overwrite=True"):
        save_model(fitted, path, {"version": 2})
    assert path.read_bytes() == original_bytes
    assert load_metadata(path)["metadata"] == {"version": 1}

    save_model(fitted, path, {"version": 2}, overwrite=True)
    assert load_metadata(path)["metadata"] == {"version": 2}


def test_rejects_unfitted_pipeline_and_bad_metadata_without_writing(tmp_path, fitted):
    with pytest.raises(NotFittedError):
        save_model(make_pipeline(), tmp_path / "unfitted.joblib")
    with pytest.raises(TypeError):
        save_model(fitted, tmp_path / "bad.joblib", {"array": np.array([1, 2])})
    assert list(tmp_path.iterdir()) == []
