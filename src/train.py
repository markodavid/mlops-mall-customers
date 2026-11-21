"""Entrenamiento y tracking de modelos con MLflow.

Entrena tres clasificadores sobre los segmentos generados con KMeans,
registra métricas, artefactos (matriz de confusión y reporte) y el modelo
en MLflow. Antes de ejecutar, levanta el servidor:

    mlflow server --host 127.0.0.1 --port 5000
"""

import inspect

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import mlflow
import mlflow.sklearn
import pandas as pd
import seaborn as sns
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src.config import (
    EXPERIMENT_NAME,
    FIGURES_DIR,
    METRICS_DIR,
    MLFLOW_TRACKING_URI,
    RANDOM_STATE,
    SEGMENTED_DATA_PATH,
    TEST_SIZE,
)
from src.features import FEATURE_COLUMNS


def load_segmented_data() -> pd.DataFrame:
    if not SEGMENTED_DATA_PATH.exists():
        raise FileNotFoundError(
            "No existe el dataset segmentado. Ejecuta primero: python -m src.clustering"
        )
    return pd.read_csv(SEGMENTED_DATA_PATH)


def save_confusion_matrix(y_test, y_pred, model_name: str):
    cm = confusion_matrix(y_test, y_pred)

    plt.figure(figsize=(7, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Matriz de confusión - {model_name}")
    plt.xlabel("Predicción")
    plt.ylabel("Valor real")
    plt.tight_layout()

    output_path = FIGURES_DIR / f"confusion_matrix_{model_name}.png"
    plt.savefig(output_path)
    plt.close()
    return output_path


def log_sklearn_model(pipeline: Pipeline, input_example: pd.DataFrame) -> None:
    """Compatible con MLflow 2.x (artifact_path) y 3.x (name)."""
    params = inspect.signature(mlflow.sklearn.log_model).parameters
    if "name" in params:
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            name="model",
            input_example=input_example,
        )
    else:  # pragma: no cover - MLflow < 2.21
        mlflow.sklearn.log_model(
            sk_model=pipeline,
            artifact_path="model",
            input_example=input_example,
        )


def train_and_log_model(model_name: str, model, X_train, X_test, y_train, y_test):
    with mlflow.start_run(run_name=model_name):
        pipeline = Pipeline(
            [
                ("scaler", StandardScaler()),
                ("model", model),
            ]
        )

        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        mlflow.log_param("model_name", model_name)
        mlflow.log_param("test_size", TEST_SIZE)
        mlflow.log_param("random_state", RANDOM_STATE)
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1", f1)

        report = classification_report(y_test, y_pred, zero_division=0, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        report_path = METRICS_DIR / f"classification_report_{model_name}.csv"
        report_df.to_csv(report_path)
        mlflow.log_artifact(str(report_path), artifact_path="metrics")

        cm_path = save_confusion_matrix(y_test, y_pred, model_name)
        mlflow.log_artifact(str(cm_path), artifact_path="figures")

        input_example = X_test.head(3)
        log_sklearn_model(pipeline, input_example)

        print(f"Modelo: {model_name}")
        print(f"  Accuracy: {accuracy:.4f} | Precision: {precision:.4f} "
              f"| Recall: {recall:.4f} | F1: {f1:.4f}")
        return {"model_name": model_name, "accuracy": accuracy, "f1": f1}


def train_models() -> pd.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    df = load_segmented_data()

    X = df[FEATURE_COLUMNS]
    y = df["customer_segment"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=TEST_SIZE,
        random_state=RANDOM_STATE,
        stratify=y,
    )

    models = {
        "logistic_regression": LogisticRegression(
            max_iter=500, random_state=RANDOM_STATE
        ),
        "random_forest": RandomForestClassifier(
            n_estimators=200, random_state=RANDOM_STATE
        ),
        "gradient_boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
    }

    results = []
    for model_name, model in models.items():
        results.append(
            train_and_log_model(model_name, model, X_train, X_test, y_train, y_test)
        )

    results_df = pd.DataFrame(results).sort_values("f1", ascending=False)
    results_df.to_csv(METRICS_DIR / "model_comparison.csv", index=False)
    print("\nComparación de modelos:")
    print(results_df.to_string(index=False))
    print(f"\nRevisa MLflow en: {MLFLOW_TRACKING_URI}")
    return results_df


if __name__ == "__main__":
    train_models()
