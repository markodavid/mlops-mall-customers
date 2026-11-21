"""Tests del pipeline de datos y features."""

import pandas as pd

from src.data import clean_column_names, preprocess_data
from src.features import FEATURE_COLUMNS, get_features, scale_features


def test_clean_column_names():
    df = pd.DataFrame(
        columns=[
            "CustomerID",
            "Gender",
            "Age",
            "Annual Income (k$)",
            "Spending Score (1-100)",
        ]
    )
    cleaned = clean_column_names(df)
    assert list(cleaned.columns) == [
        "customerid",
        "gender",
        "age",
        "annual_income_k$",
        "spending_score_1_100",
    ]


def test_preprocess_data_with_real_dataset():
    df = preprocess_data()
    assert list(df.columns) == FEATURE_COLUMNS
    assert len(df) > 0
    assert df["gender"].isin([0, 1]).all()
    assert df.isnull().sum().sum() == 0


def test_features_and_scaling():
    df = pd.DataFrame(
        {
            "gender": [1, 0, 1],
            "age": [19, 35, 50],
            "annual_income_k$": [15, 60, 120],
            "spending_score_1_100": [39, 50, 80],
        }
    )
    X = get_features(df)
    assert list(X.columns) == FEATURE_COLUMNS

    X_scaled, scaler = scale_features(X)
    assert X_scaled.shape == (3, 4)
    assert scaler.mean_.shape == (4,)
