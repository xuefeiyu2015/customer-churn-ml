"""Feature cleaning and preprocessing for the Telco churn model.

Two stages, both reproduced from ``notebooks/churn_analysis.ipynb``:

1. ``clean_features`` – stateless, row-wise cleaning of raw CSV-style rows
   (column/value normalization, TotalCharges rule, Yes/No -> 1/0, validation,
   feature selection). Top-level function so it can be wrapped in
   ``sklearn.preprocessing.FunctionTransformer`` and pickled with joblib.
2. ``build_preprocessor`` – the notebook's unfitted ``ColumnTransformer``
   (one-hot encoding + scaling). Its statistics are learned only when the
   pipeline is fit, i.e. on training data only.

Target handling is kept separate in ``prepare_target`` and is used in training
only.

Unknown-category policy (simplified): unseen values in categorical/service
columns are passed through and ignored by ``OneHotEncoder(handle_unknown=
"ignore")`` (all-zero block). For columns encoded with ``drop="if_binary"``
(e.g. ``gender``) an unseen value is therefore indistinguishable from the
dropped reference category. Missing values and non-numeric numerics are
rejected. No category registry or drift monitoring is implemented.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler

ID_COLUMN = "customerid"
TARGET_COLUMN = "churn"

YES_NO = {"yes": 1, "no": 0}

FE_CATEGORICAL = ["gender", "internetservice", "contract", "paymentmethod"]
FE_NUMERIC = ["tenure", "monthlycharges", "totalcharges"]
FE_BINARY = ["seniorcitizen", "partner", "dependents", "phoneservice", "paperlessbilling"]
FE_SERVICE = [
    "multiplelines",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
]

# Yes/No string columns mapped to 1/0 (seniorcitizen is already 0/1 in the raw data).
YES_NO_COLUMNS = ["partner", "dependents", "phoneservice", "paperlessbilling"]
STRING_FEATURES = FE_CATEGORICAL + FE_SERVICE

# Same column order as the notebook's X (raw CSV order minus customerID/Churn).
FEATURE_COLUMNS = [
    "gender",
    "seniorcitizen",
    "partner",
    "dependents",
    "tenure",
    "phoneservice",
    "multiplelines",
    "internetservice",
    "onlinesecurity",
    "onlinebackup",
    "deviceprotection",
    "techsupport",
    "streamingtv",
    "streamingmovies",
    "contract",
    "paperlessbilling",
    "paymentmethod",
    "monthlycharges",
    "totalcharges",
]

_BLANK_PATTERN = r"^[\s_]*$"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Lower-case column names and string values, replacing ' ' with '_' (notebook cell 15).

    Blank / underscore-only strings become NaN (notebook ``clean_churn_data``).
    Idempotent, so already-normalized input passes through unchanged.
    """
    out = df.copy()
    out.columns = out.columns.str.lower().str.replace(" ", "_")

    duplicated = out.columns[out.columns.duplicated()].tolist()
    if duplicated:
        raise ValueError(f"Duplicate column names after normalization: {duplicated}")

    str_cols = out.select_dtypes(include=["object", "string"]).columns
    for c in str_cols:
        out[c] = out[c].str.lower().str.replace(" ", "_")
    out[str_cols] = out[str_cols].replace(_BLANK_PATTERN, np.nan, regex=True)
    return out


def _to_numeric_strict(s: pd.Series, name: str) -> pd.Series:
    """Convert to numeric; raise on values that are present but not numeric."""
    converted = pd.to_numeric(s, errors="coerce")
    invalid = converted.isna() & s.notna()
    if invalid.any():
        raise ValueError(f"Invalid numeric values in {name}: {s[invalid].unique().tolist()}")
    return converted


def clean_features(X: pd.DataFrame) -> pd.DataFrame:
    """Clean raw customer rows into the model's feature frame.

    Accepts original CSV column names (``customerID``, ``TotalCharges``, ...)
    or already-normalized names. ``customerID`` and ``Churn`` may be present
    and are dropped. Returns a new DataFrame with exactly ``FEATURE_COLUMNS``
    in that order, preserving the input index. The input is not modified.

    Raises ``ValueError`` on missing required columns, missing values,
    non-numeric numerics, non Yes/No binary values, and missing TotalCharges
    for customers with tenure > 0.
    """
    out = normalize_columns(X)

    missing_cols = [c for c in FEATURE_COLUMNS if c not in out.columns]
    if missing_cols:
        raise ValueError(f"Missing required feature columns: {missing_cols}")
    out = out[FEATURE_COLUMNS].copy()

    # Missing values are rejected everywhere except TotalCharges (handled below).
    required = [c for c in FEATURE_COLUMNS if c != "totalcharges"]
    na_counts = out[required].isna().sum()
    if na_counts.any():
        raise ValueError(f"Missing values in required features: {na_counts[na_counts > 0].to_dict()}")

    for c in ("tenure", "monthlycharges", "seniorcitizen"):
        out[c] = _to_numeric_strict(out[c], c)

    if not out["seniorcitizen"].isin([0, 1]).all():
        raise ValueError("seniorcitizen must be 0 or 1")
    out["seniorcitizen"] = out["seniorcitizen"].astype("int64")

    for c in YES_NO_COLUMNS:
        invalid = ~out[c].isin(YES_NO)
        if invalid.any():
            raise ValueError(f"Unexpected values in {c}: {out.loc[invalid, c].unique().tolist()}")
        out[c] = out[c].map(YES_NO).astype("int64")

    # TotalCharges rule (notebook cell 22): blank & tenure == 0 -> 0, blank & tenure > 0 -> error.
    total = _to_numeric_strict(out["totalcharges"], "totalcharges")
    missing_total = total.isna()
    bad = missing_total & (out["tenure"] != 0)
    if bad.any():
        raise ValueError(
            f"Found {int(bad.sum())} customers with missing TotalCharges and tenure > 0"
        )
    out["totalcharges"] = total.mask(missing_total, 0.0).astype("float64")

    return out


def prepare_target(df: pd.DataFrame) -> pd.Series:
    """Map the raw ``Churn`` column (Yes/No) to 1/0. Training only."""
    normalized = normalize_columns(df)
    if TARGET_COLUMN not in normalized.columns:
        raise ValueError(f"Missing target column: {TARGET_COLUMN!r}")
    y = normalized[TARGET_COLUMN]
    invalid = ~y.isin(YES_NO)
    if invalid.any():
        raise ValueError(f"Unexpected churn values: {y[invalid].unique().tolist()}")
    return y.map(YES_NO).astype("int64").rename(TARGET_COLUMN)


def build_preprocessor() -> ColumnTransformer:
    """Unfitted ColumnTransformer, identical to the notebook (40 output features)."""
    return ColumnTransformer(
        transformers=[
            ("cat", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), FE_CATEGORICAL),
            ("num", StandardScaler(), FE_NUMERIC),
            ("bin", "passthrough", FE_BINARY),
            ("ser", OneHotEncoder(handle_unknown="ignore", drop="if_binary"), FE_SERVICE),
        ],
        verbose_feature_names_out=False,
    )
