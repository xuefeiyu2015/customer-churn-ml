import pickle

import numpy as np
import pandas as pd
import pytest
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer

from customer_churn_ml.preprocessing import (
    FEATURE_COLUMNS,
    build_preprocessor,
    clean_features,
    prepare_target,
)

SERVICES = ["OnlineSecurity", "OnlineBackup", "DeviceProtection",
            "TechSupport", "StreamingTV", "StreamingMovies"]

# Feature names printed by the notebook (cell 53) for the full training set.
NOTEBOOK_FEATURE_NAMES = [
    "gender_male", "internetservice_dsl", "internetservice_fiber_optic", "internetservice_no",
    "contract_month-to-month", "contract_one_year", "contract_two_year",
    "paymentmethod_bank_transfer_(automatic)", "paymentmethod_credit_card_(automatic)",
    "paymentmethod_electronic_check", "paymentmethod_mailed_check",
    "tenure", "monthlycharges", "totalcharges",
    "seniorcitizen", "partner", "dependents", "phoneservice", "paperlessbilling",
    "multiplelines_no", "multiplelines_no_phone_service", "multiplelines_yes",
    "onlinesecurity_no", "onlinesecurity_no_internet_service", "onlinesecurity_yes",
    "onlinebackup_no", "onlinebackup_no_internet_service", "onlinebackup_yes",
    "deviceprotection_no", "deviceprotection_no_internet_service", "deviceprotection_yes",
    "techsupport_no", "techsupport_no_internet_service", "techsupport_yes",
    "streamingtv_no", "streamingtv_no_internet_service", "streamingtv_yes",
    "streamingmovies_no", "streamingmovies_no_internet_service", "streamingmovies_yes",
]


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Four raw CSV-style rows covering every category in the real dataset."""
    df = pd.DataFrame({
        "customerID": ["0001-AAAA", "0002-BBBB", "0003-CCCC", "0004-DDDD"],
        "gender": ["Female", "Male", "Male", "Female"],
        "SeniorCitizen": [0, 1, 0, 0],
        "Partner": ["Yes", "No", "No", "Yes"],
        "Dependents": ["No", "No", "Yes", "No"],
        "tenure": [1, 34, 0, 45],
        "PhoneService": ["No", "Yes", "Yes", "Yes"],
        "MultipleLines": ["No phone service", "No", "Yes", "No"],
        "InternetService": ["DSL", "Fiber optic", "No", "DSL"],
        **{s: ["No", "Yes", "No internet service", "No"] for s in SERVICES},
        "Contract": ["Month-to-month", "One year", "Two year", "Month-to-month"],
        "PaperlessBilling": ["Yes", "No", "No", "Yes"],
        "PaymentMethod": ["Electronic check", "Mailed check",
                          "Bank transfer (automatic)", "Credit card (automatic)"],
        "MonthlyCharges": [29.85, 56.95, 20.25, 42.30],
        "TotalCharges": ["29.85", "1889.5", " ", "1840.75"],
        "Churn": ["No", "No", "Yes", "Yes"],
    })
    df.index = [10, 11, 12, 13]  # non-default index must be preserved
    return df


def test_clean_features_raw_csv_input(raw_df):
    out = clean_features(raw_df)

    assert list(out.columns) == FEATURE_COLUMNS
    assert "customerid" not in out.columns and "churn" not in out.columns
    assert list(out.index) == list(raw_df.index)
    assert out.notna().all().all()
    assert out.loc[11, "internetservice"] == "fiber_optic"
    assert out.loc[12, "paymentmethod"] == "bank_transfer_(automatic)"
    assert out.loc[10, "multiplelines"] == "no_phone_service"
    assert out["totalcharges"].dtype == "float64"
    assert out.loc[11, "totalcharges"] == pytest.approx(1889.5)


def test_binary_features_mapped_to_int(raw_df):
    out = clean_features(raw_df)
    assert out["partner"].tolist() == [1, 0, 0, 1]
    assert out["phoneservice"].tolist() == [0, 1, 1, 1]
    for c in ["seniorcitizen", "partner", "dependents", "phoneservice", "paperlessbilling"]:
        assert out[c].dtype == "int64"


def test_invalid_binary_value_raises(raw_df):
    raw_df.loc[10, "Partner"] = "Maybe"
    with pytest.raises(ValueError, match="partner"):
        clean_features(raw_df)


def test_totalcharges_blank_with_zero_tenure_becomes_zero(raw_df):
    assert clean_features(raw_df).loc[12, "totalcharges"] == 0.0


@pytest.mark.parametrize("value", [" ", "", np.nan])
def test_totalcharges_missing_with_positive_tenure_raises(raw_df, value):
    raw_df.loc[13, "TotalCharges"] = value
    with pytest.raises(ValueError, match="tenure > 0"):
        clean_features(raw_df)


def test_totalcharges_non_numeric_raises(raw_df):
    raw_df.loc[12, "TotalCharges"] = "abc"  # invalid even though tenure == 0
    with pytest.raises(ValueError, match="Invalid numeric"):
        clean_features(raw_df)


def test_missing_required_value_or_column_raises(raw_df):
    with pytest.raises(ValueError, match="Missing values"):
        clean_features(raw_df.assign(Contract=["Month-to-month", " ", "Two year", "One year"]))
    with pytest.raises(ValueError, match="Missing required feature columns"):
        clean_features(raw_df.drop(columns="tenure"))


def test_feature_target_separation(raw_df):
    features_only = clean_features(raw_df.drop(columns=["Churn", "customerID"]))
    pd.testing.assert_frame_equal(features_only, clean_features(raw_df))

    y = prepare_target(raw_df)
    assert y.name == "churn"
    assert y.tolist() == [0, 0, 1, 1]
    assert list(y.index) == list(raw_df.index)

    with pytest.raises(ValueError, match="Unexpected churn"):
        prepare_target(raw_df.assign(Churn=["No", "Yes", "Unknown", "No"]))


def test_inputs_not_mutated(raw_df):
    before = raw_df.copy()
    clean_features(raw_df)
    prepare_target(raw_df)
    pd.testing.assert_frame_equal(raw_df, before)


def test_preprocessor_output_matches_notebook_feature_order(raw_df):
    prep = build_preprocessor()
    Xt = prep.fit_transform(clean_features(raw_df))
    assert Xt.shape == (4, 40)
    assert list(prep.get_feature_names_out()) == NOTEBOOK_FEATURE_NAMES


def test_unknown_categories_are_ignored(raw_df):
    prep = build_preprocessor().fit(clean_features(raw_df))
    new = raw_df.iloc[[0]].copy()
    new["PaymentMethod"] = "Crypto"
    new["Contract"] = "Weekly"

    with pytest.warns(UserWarning, match="unknown categories"):
        Xt = prep.transform(clean_features(new))
    names = list(prep.get_feature_names_out())
    row = pd.Series(np.asarray(Xt)[0], index=names)

    assert Xt.shape == (1, 40)
    assert (row.filter(like="paymentmethod_") == 0).all()
    assert (row.filter(like="contract_") == 0).all()


def test_clean_features_in_function_transformer_survives_pickle(raw_df):
    pipe = Pipeline([
        ("clean", FunctionTransformer(clean_features)),
        ("prep", build_preprocessor()),
    ]).fit(raw_df)
    restored = pickle.loads(pickle.dumps(pipe))
    np.testing.assert_array_equal(restored.transform(raw_df), pipe.transform(raw_df))
