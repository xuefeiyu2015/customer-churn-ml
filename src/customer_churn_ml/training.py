"""Reproducible training of the selected churn model.

Reproduces the notebook's final workflow (``notebooks/churn_analysis.ipynb``):
stratified 80/20 split (random_state=42) -> fit the complete Pipeline
(clean_features -> ColumnTransformer -> RandomForest) on the training split
only -> evaluate once on the untouched holdout -> save -> reload and verify.

All learned preprocessing statistics (scaler means/stds, one-hot categories)
live inside the Pipeline and are fitted by ``Pipeline.fit`` on the training
split only; the holdout is only ever passed to ``predict_proba``.

Usage::

    uv run python -m customer_churn_ml.training \
        --data data/WA_Fn-UseC_-Telco-Customer-Churn.csv \
        --model-out models/churn_random_forest.joblib
"""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from customer_churn_ml.evaluation import (
    CONTACT_CAPACITY,
    REFERENCE_POPULATION,
    compute_k,
    compute_metrics,
    positive_class_proba,
)
from customer_churn_ml.persistence import load_model, save_model
from customer_churn_ml.preprocessing import (
    TARGET_COLUMN,
    build_preprocessor,
    clean_features,
    prepare_target,
)

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA_PATH = PROJECT_ROOT / "data" / "WA_Fn-UseC_-Telco-Customer-Churn.csv"
DEFAULT_MODEL_PATH = PROJECT_ROOT / "models" / "churn_random_forest.joblib"

RANDOM_STATE = 42
TEST_SIZE = 0.2
CV_FOLDS = 5

# Final model selected in the notebook (cells 64-71): best mean CV average precision.
RF_PARAMS: dict[str, Any] = {
    "n_estimators": 300,
    "max_depth": 10,
    "min_samples_leaf": 20,
    "class_weight": None,
    "random_state": RANDOM_STATE,
    "n_jobs": 1,
}


@dataclass(frozen=True)
class TrainingResult:
    pipeline: Pipeline
    holdout_metrics: dict[str, float | int]
    oof_metrics: dict[str, float | int] | None
    train_index: pd.Index
    test_index: pd.Index
    model_path: Path
    metadata_path: Path
    metadata: dict[str, Any]


# ---------------------------------------------------------------- data


def load_raw(path: str | Path) -> pd.DataFrame:
    """Read the raw CSV and run the notebook's basic integrity checks (cell 4)."""
    path = Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"Data file not found: {path}")
    raw = pd.read_csv(path)
    if raw.empty:
        raise ValueError(f"Data file is empty: {path}")

    lower = {c.lower(): c for c in raw.columns}
    missing = [c for c in ("customerid", TARGET_COLUMN) if c not in lower]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")
    ids = raw[lower["customerid"]]
    if ids.isna().any():
        raise ValueError("Missing customerID detected")
    if ids.duplicated().any():
        raise ValueError(f"Duplicate customerID values: {int(ids.duplicated().sum())}")
    return raw


def split_features_target(raw: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    """Raw feature rows (target column removed, cleaning deferred to the Pipeline) and 0/1 target."""
    y = prepare_target(raw)
    X = raw.drop(columns=[c for c in raw.columns if c.lower() == TARGET_COLUMN])
    return X, y


def split_data(
    X: pd.DataFrame,
    y: pd.Series,
    test_size: float = TEST_SIZE,
    random_state: int = RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Stratified split identical to notebook cell 50."""
    return train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)


def file_sha256(path: str | Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# ---------------------------------------------------------------- model


def build_model() -> RandomForestClassifier:
    return RandomForestClassifier(**RF_PARAMS)


def build_pipeline(model=None) -> Pipeline:
    """Complete unfitted Pipeline: raw rows -> clean -> encode/scale -> model."""
    return Pipeline([
        ("clean", FunctionTransformer(clean_features)),
        ("prep", build_preprocessor()),
        ("model", clone(model) if model is not None else build_model()),
    ])


# ---------------------------------------------------------------- evaluation


def evaluate_pipeline(pipeline: Pipeline, X: pd.DataFrame, y: pd.Series) -> dict[str, float | int]:
    """Metrics on an evaluation set; K scaled by 500/7043 to the set size."""
    scores = positive_class_proba(pipeline, X)
    return compute_metrics(y, scores, compute_k(len(y)))


def compute_oof_metrics(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = CV_FOLDS,
    random_state: int = RANDOM_STATE,
) -> dict[str, float | int]:
    """Out-of-fold metrics on the training split (notebook cells 76-83; K=400 on real data).

    The whole Pipeline is refit inside each fold, so no fold sees statistics
    from its own validation rows. The holdout is not used.
    """
    cv = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=random_state)
    proba = cross_val_predict(build_pipeline(), X_train, y_train, cv=cv, method="predict_proba")
    classes = np.unique(y_train)  # cross_val_predict orders columns by sorted classes
    scores = proba[:, int(np.flatnonzero(classes == 1)[0])]
    return compute_metrics(y_train, scores, compute_k(len(y_train)))


# ---------------------------------------------------------------- workflow


def run_training(
    data_path: str | Path = DEFAULT_DATA_PATH,
    model_path: str | Path = DEFAULT_MODEL_PATH,
    *,
    overwrite: bool = False,
    include_oof: bool = True,
) -> TrainingResult:
    """Train, evaluate, save, reload and verify the final model."""
    raw = load_raw(data_path)
    X, y = split_features_target(raw)
    X_train, X_test, y_train, y_test = split_data(X, y)

    pipeline = build_pipeline().fit(X_train, y_train)  # training split only
    holdout_metrics = evaluate_pipeline(pipeline, X_test, y_test)
    oof_metrics = compute_oof_metrics(X_train, y_train) if include_oof else None
    # The FunctionTransformer step has no feature names; read them from the ColumnTransformer.
    encoded_names = pipeline.named_steps["prep"].get_feature_names_out().tolist()

    metadata = {
        "model": "RandomForestClassifier",
        "model_params": RF_PARAMS,
        "split": {
            "test_size": TEST_SIZE,
            "random_state": RANDOM_STATE,
            "stratify": TARGET_COLUMN,
            "n_train": int(len(y_train)),
            "n_test": int(len(y_test)),
        },
        "k_rule": {"capacity": CONTACT_CAPACITY, "population": REFERENCE_POPULATION},
        "features": {
            "n_encoded": len(encoded_names),
            "encoded_names": encoded_names,
        },
        "holdout_metrics": holdout_metrics,
        "oof_metrics": oof_metrics,
        "data": {"file": Path(data_path).name, "sha256": file_sha256(data_path), "n_rows": len(raw)},
    }
    saved_model, saved_meta = save_model(pipeline, model_path, metadata, overwrite=overwrite)

    reloaded = load_model(saved_model)
    if not np.array_equal(
        positive_class_proba(reloaded, X_test), positive_class_proba(pipeline, X_test)
    ):
        raise RuntimeError(f"Reloaded model predictions differ from the fitted pipeline: {saved_model}")

    return TrainingResult(
        pipeline=pipeline,
        holdout_metrics=holdout_metrics,
        oof_metrics=oof_metrics,
        train_index=X_train.index,
        test_index=X_test.index,
        model_path=saved_model,
        metadata_path=saved_meta,
        metadata=metadata,
    )


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Train the churn RandomForest pipeline.")
    parser.add_argument("--data", type=Path, default=DEFAULT_DATA_PATH)
    parser.add_argument("--model-out", type=Path, default=DEFAULT_MODEL_PATH)
    parser.add_argument("--overwrite", action="store_true", help="replace existing model artifacts")
    parser.add_argument("--no-oof", action="store_true", help="skip out-of-fold evaluation")
    args = parser.parse_args(argv)

    result = run_training(args.data, args.model_out, overwrite=args.overwrite, include_oof=not args.no_oof)
    print(json.dumps(
        {"holdout": result.holdout_metrics, "oof": result.oof_metrics,
         "model_path": str(result.model_path), "metadata_path": str(result.metadata_path)},
        indent=2,
    ))


if __name__ == "__main__":
    main()
