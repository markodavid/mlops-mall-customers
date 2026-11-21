"""Definición de features y escalado.

FEATURE_COLUMNS es la única fuente de verdad del orden de columnas que espera
el modelo. Se usa al entrenar, al predecir desde src/predict.py y en la API.
"""

import pandas as pd
from sklearn.preprocessing import StandardScaler

FEATURE_COLUMNS = [
    "gender",
    "age",
    "annual_income_k$",
    "spending_score_1_100",
]


def get_features(df: pd.DataFrame) -> pd.DataFrame:
    """Devuelve solo las columnas usadas por el modelo, en el orden correcto."""
    return df[FEATURE_COLUMNS].copy()


def scale_features(X: pd.DataFrame):
    """Escala las features con StandardScaler (media 0, desviación 1)."""
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler
