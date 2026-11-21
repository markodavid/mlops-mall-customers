"""Tests de la API con TestClient (no requieren MLflow en ejecución)."""

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.main as api_main
from api.main import app


class DummyModel:
    """Modelo falso: siempre predice el segmento 2."""

    def predict(self, df):
        return np.array([2] * len(df))


@pytest.fixture
def client():
    with TestClient(app) as test_client:
        yield test_client


def test_root(client):
    response = client.get("/")
    assert response.status_code == 200
    assert "message" in response.json()


def test_health(client):
    response = client.get("/health")
    assert response.status_code == 200
    body = response.json()
    assert body["status"] == "ok"
    assert "model_loaded" in body


def test_metrics_exposes_prometheus_format(client):
    response = client.get("/metrics")
    assert response.status_code == 200
    assert "mall_customer_prediction_count_total" in response.text


def test_predict_invalid_gender(client):
    response = client.post(
        "/predict",
        json={
            "gender": "otro",
            "age": 30,
            "annual_income_k": 60,
            "spending_score": 50,
        },
    )
    assert response.status_code == 400


def test_predict_validation_error(client):
    response = client.post(
        "/predict",
        json={
            "gender": "Male",
            "age": 5,  # fuera de rango
            "annual_income_k": 60,
            "spending_score": 50,
        },
    )
    assert response.status_code == 422


def test_predict_success_with_dummy_model(client, monkeypatch, tmp_path):
    monkeypatch.setattr(api_main, "model", DummyModel())
    monkeypatch.setattr(api_main, "PREDICTIONS_LOG_PATH", tmp_path / "log.csv")

    response = client.post(
        "/predict",
        json={
            "gender": "Male",
            "age": 28,
            "annual_income_k": 60,
            "spending_score": 80,
        },
    )

    assert response.status_code == 200
    body = response.json()
    assert body["customer_segment"] == 2
    assert body["latency_seconds"] >= 0
    assert (tmp_path / "log.csv").exists()


def test_predict_without_model_returns_503(client, monkeypatch):
    monkeypatch.setattr(api_main, "model", None)

    response = client.post(
        "/predict",
        json={
            "gender": "Female",
            "age": 40,
            "annual_income_k": 80,
            "spending_score": 60,
        },
    )

    assert response.status_code == 503
