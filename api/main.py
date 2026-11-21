"""API REST con FastAPI: predicción de segmentos + métricas Prometheus."""

import time
from contextlib import asynccontextmanager
from datetime import datetime, timezone

import mlflow
import mlflow.pyfunc
import pandas as pd
from fastapi import FastAPI, HTTPException, Response, status
from prometheus_client import (
    CONTENT_TYPE_LATEST,
    Counter,
    Histogram,
    generate_latest,
)
from pydantic import BaseModel, Field

from src.config import (
    LOAD_MODEL_ON_STARTUP,
    MLFLOW_TRACKING_URI,
    MODEL_ALIAS,
    MONITORING_DIR,
    PREDICTIONS_LOG_PATH,
    REGISTERED_MODEL_NAME,
)
from src.features import FEATURE_COLUMNS

# =========================
# MÉTRICAS PROMETHEUS
# =========================

PREDICTION_COUNT = Counter(
    "mall_customer_prediction_count",
    "Número total de predicciones de segmentos de clientes",
)

PREDICTION_LATENCY = Histogram(
    "mall_customer_prediction_latency_seconds",
    "Latencia de predicción del modelo en segundos",
)

SEGMENT_COUNTER = Counter(
    "mall_customer_segment_predictions_total",
    "Predicciones por segmento",
    ["segment"],
)

# =========================
# MODELO GLOBAL
# =========================

model = None


def load_champion_model() -> None:
    """Carga el modelo 'champion' desde MLflow sin tumbar la API si falla."""
    global model

    try:
        mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
        model_uri = f"models:/{REGISTERED_MODEL_NAME}@{MODEL_ALIAS}"
        model = mlflow.pyfunc.load_model(model_uri)
        MONITORING_DIR.mkdir(parents=True, exist_ok=True)
        print(f"Modelo cargado correctamente desde {model_uri}")
    except Exception as exc:  # noqa: BLE001 - la API debe seguir viva
        model = None
        print(
            "[WARN] No se pudo cargar el modelo champion. "
            f"La API responderá 503 en /predict. Detalle: {exc}"
        )


@asynccontextmanager
async def lifespan(_app: FastAPI):
    if LOAD_MODEL_ON_STARTUP:
        load_champion_model()
    else:
        print("Carga automática del modelo desactivada (LOAD_MODEL_ON_STARTUP=0)")
    yield


app = FastAPI(
    title="Mall Customers Classification API",
    version="1.0.0",
    description="Clasifica clientes en segmentos usando el modelo champion de MLflow.",
    lifespan=lifespan,
)


# =========================
# REQUEST / RESPONSE
# =========================


class PredictionRequest(BaseModel):
    gender: str = Field(..., examples=["Male"], description="Male o Female")
    age: int = Field(..., ge=18, le=100, examples=[28])
    annual_income_k: int = Field(..., ge=0, le=200, examples=[60])
    spending_score: int = Field(..., ge=1, le=100, examples=[80])


# =========================
# ENDPOINTS
# =========================


@app.get("/")
def root():
    return {"message": "Mall Customers API activa"}


@app.get("/health")
def health():
    return {
        "status": "ok",
        "model": REGISTERED_MODEL_NAME,
        "alias": MODEL_ALIAS,
        "model_loaded": model is not None,
    }


@app.post("/predict")
def predict(request: PredictionRequest):
    start_time = time.time()

    gender_value = request.gender.strip().lower()
    if gender_value not in ["male", "female"]:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="gender debe ser Male o Female",
        )

    if model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "El modelo todavía no está disponible. Verifica que MLflow esté "
                "corriendo y que el modelo tenga el alias configurado."
            ),
        )

    gender_encoded = 1 if gender_value == "male" else 0

    input_df = pd.DataFrame(
        [
            {
                "gender": int(gender_encoded),
                "age": int(request.age),
                "annual_income_k$": int(request.annual_income_k),
                "spending_score_1_100": int(request.spending_score),
            }
        ]
    ).astype(
        {
            "gender": "int64",
            "age": "int64",
            "annual_income_k$": "int64",
            "spending_score_1_100": "int64",
        }
    )

    input_df = input_df[FEATURE_COLUMNS]

    prediction = int(model.predict(input_df)[0])
    latency = time.time() - start_time

    PREDICTION_COUNT.inc()
    PREDICTION_LATENCY.observe(latency)
    SEGMENT_COUNTER.labels(segment=str(prediction)).inc()

    log_prediction(input_df=input_df, prediction=prediction, latency=latency)

    return {
        "customer_segment": prediction,
        "message": f"El cliente pertenece al segmento {prediction}",
        "latency_seconds": round(latency, 6),
        "timestamp": datetime.now(timezone.utc).isoformat(),
    }


@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type=CONTENT_TYPE_LATEST)


# =========================
# LOG DE PREDICCIONES
# =========================


def log_prediction(input_df: pd.DataFrame, prediction: int, latency: float) -> None:
    log_file = PREDICTIONS_LOG_PATH
    log_file.parent.mkdir(parents=True, exist_ok=True)

    row = input_df.copy()
    row["customer_segment"] = prediction
    row["latency_seconds"] = latency
    row["timestamp"] = datetime.now(timezone.utc).isoformat()

    row.to_csv(
        log_file,
        mode="a" if log_file.exists() else "w",
        header=not log_file.exists(),
        index=False,
    )
