"""Prueba de predicción local cargando el modelo 'champion' desde MLflow."""

import mlflow
import mlflow.pyfunc
import pandas as pd

from src.config import MLFLOW_TRACKING_URI, MODEL_ALIAS, REGISTERED_MODEL_NAME


def load_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
    return mlflow.pyfunc.load_model(model_uri)


def predict_customer() -> int:
    model = load_model()

    input_df = pd.DataFrame(
        [
            {
                "gender": 1,
                "age": 35,
                "annual_income_k$": 120,
                "spending_score_1_100": 50,
            }
        ]
    )

    prediction = int(model.predict(input_df)[0])
    print("Segmento predicho:", prediction)
    return prediction


if __name__ == "__main__":
    predict_customer()
