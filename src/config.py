"""Configuración centralizada del proyecto.

Todas las rutas, nombres y parámetros viven aquí para que el resto del código
no dependa de valores "hardcodeados". Los valores se pueden sobreescribir con
variables de entorno (ver .env.example).
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# En Windows la consola usa cp1252 por defecto y MLflow imprime emojis al
# cerrar un run (UnicodeEncodeError). Forzamos UTF-8 en stdout/stderr.
for _stream in (sys.stdout, sys.stderr):
    if hasattr(_stream, "reconfigure"):
        try:
            _stream.reconfigure(encoding="utf-8", errors="replace")
        except (OSError, ValueError):  # pragma: no cover
            pass


ROOT_DIR = Path(__file__).resolve().parents[1]

load_dotenv(ROOT_DIR / ".env")

DATA_DIR = ROOT_DIR / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MONITORING_DIR = DATA_DIR / "monitoring"

REPORTS_DIR = ROOT_DIR / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
METRICS_DIR = REPORTS_DIR / "metrics"
MODELS_DIR = ROOT_DIR / "models"

RAW_DATA_PATH = RAW_DATA_DIR / "Mall_Customers.csv"
PROCESSED_DATA_PATH = PROCESSED_DATA_DIR / "mall_customers_processed.csv"
SEGMENTED_DATA_PATH = PROCESSED_DATA_DIR / "mall_customers_segmented.csv"
SEGMENT_SUMMARY_PATH = METRICS_DIR / "segment_summary.csv"
PREDICTIONS_LOG_PATH = MONITORING_DIR / "api_predictions_log.csv"

MLFLOW_TRACKING_URI = os.getenv("MLFLOW_TRACKING_URI", "http://127.0.0.1:5000")
EXPERIMENT_NAME = os.getenv("EXPERIMENT_NAME", "mall-customers-classification")
REGISTERED_MODEL_NAME = os.getenv(
    "REGISTERED_MODEL_NAME", "mall-customer-segment-classifier"
)
MODEL_ALIAS = os.getenv("MODEL_ALIAS", "champion")

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

# En tests se desactiva la carga real del modelo desde MLflow.
LOAD_MODEL_ON_STARTUP = os.getenv("LOAD_MODEL_ON_STARTUP", "1").lower() not in {
    "0",
    "false",
    "no",
}

RANDOM_STATE = 42
TEST_SIZE = 0.2
N_CLUSTERS = 4
