"""Registra en el Model Registry el mejor run (por F1) y le asigna el alias."""

import mlflow
from mlflow.tracking import MlflowClient

from src.config import (
    EXPERIMENT_NAME,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    REGISTERED_MODEL_NAME,
)


def register_best_model():
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    client = MlflowClient(tracking_uri=MLFLOW_TRACKING_URI)

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment is None:
        raise ValueError(
            f"No existe el experimento '{EXPERIMENT_NAME}'. "
            "Ejecuta primero: python -m src.train"
        )

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        order_by=["metrics.f1 DESC"],
        max_results=1,
    )

    if not runs:
        raise ValueError("No hay runs disponibles en el experimento.")

    best_run = runs[0]
    run_id = best_run.info.run_id

    print(f"Mejor run: {run_id}")
    print(f"  F1: {best_run.data.metrics.get('f1', float('nan')):.4f}")
    print(f"  Accuracy: {best_run.data.metrics.get('accuracy', float('nan')):.4f}")

    model_uri = f"runs:/{run_id}/model"
    result = mlflow.register_model(model_uri=model_uri, name=REGISTERED_MODEL_NAME)

    client.set_registered_model_alias(
        name=REGISTERED_MODEL_NAME,
        alias=MODEL_ALIAS,
        version=result.version,
    )
    client.set_model_version_tag(
        name=REGISTERED_MODEL_NAME,
        version=result.version,
        key="source_run_id",
        value=run_id,
    )

    print("Modelo registrado correctamente")
    print("Run ID:", run_id)
    print("Modelo:", REGISTERED_MODEL_NAME)
    print("Versión:", result.version)
    print("Alias:", MODEL_ALIAS)
    return result


if __name__ == "__main__":
    register_best_model()
