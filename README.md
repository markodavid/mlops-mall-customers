# MLOps Mall Customers 🛍️

Proyecto MLOps de extremo a extremo: segmentación de clientes con **KMeans**, entrenamiento de clasificadores con **scikit-learn**, tracking y registro de modelos con **MLflow**, API con **FastAPI**, frontend con **Streamlit** y monitoreo con **Prometheus + Grafana**.

Basado en el tutorial *"Cómo construir un proyecto MLOps completo con Python, MLflow, FastAPI, Streamlit y Grafana"* de Juan Manuel Núñez Velasco, con ajustes de robustez (MLflow 3.x, ejecución headless en CI, API tolerante a fallos de MLflow y dashboard de Grafana aprovisionado automáticamente).

---

## Arquitectura

```
Cliente → Streamlit (8501) → FastAPI (8000) → MLflow Model Registry (5000)
                                   │
                                   ├→ /metrics (Prometheus) → Grafana (3000)
                                   └→ data/monitoring/api_predictions_log.csv
```

| Componente | Puerto | Descripción |
|---|---|---|
| MLflow | 5000 | Tracking de experimentos y Model Registry |
| FastAPI | 8000 | Endpoints `/predict`, `/health`, `/metrics` |
| Streamlit | 8501 | Interfaz para probar el modelo |
| Prometheus | 9090 | Recolección de métricas |
| Grafana | 3000 | Dashboards de monitoreo |

## Estructura

```
mlops-mall-customers/
├── api/
│   └── main.py                  # FastAPI + métricas Prometheus + logging
├── app/
│   └── streamlit_app.py         # Frontend Streamlit
├── data/
│   ├── raw/Mall_Customers.csv   # Dataset original (versionado, lo usa CI)
│   ├── processed/               # Datos limpios y segmentados (generados)
│   └── monitoring/              # Log de predicciones (generado)
├── docker/
│   ├── prometheus.yml
│   └── grafana/                 # Provisioning + dashboard listo
├── reports/
│   ├── figures/                 # Gráficas y matrices de confusión
│   └── metrics/                 # Reportes y resumen de segmentos
├── src/
│   ├── config.py                # Configuración centralizada
│   ├── data.py                  # Carga y limpieza
│   ├── features.py              # FEATURE_COLUMNS + escalado
│   ├── clustering.py            # KMeans + PCA + visualizaciones
│   ├── train.py                 # Entrenamiento + tracking en MLflow
│   ├── register_model.py        # Registro del mejor modelo (alias champion)
│   └── predict.py               # Prueba de predicción local
├── tests/                       # pytest (API + datos)
├── .github/workflows/ci.yml     # CI: compila, preprocesa, segmenta, testea
├── docker-compose.yml
├── requirements.txt
└── README.md
```

## Requisitos

- Python 3.11 o 3.12
- Docker Desktop (para Prometheus y Grafana)
- Git

## Puesta en marcha

### 1. Entorno virtual e instalación

```powershell
python -m venv .venv
.venv\Scripts\activate
pip install --upgrade pip
pip install -r requirements.txt
```

En macOS/Linux:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### 2. Configuración

```powershell
Copy-Item .env.example .env
```

El archivo `.env` es opcional; sin él se usan los valores por defecto de `src/config.py`.

### 3. Preprocesamiento

```powershell
python -m src.data
```

Genera `data/processed/mall_customers_processed.csv` con las columnas `gender`, `age`, `annual_income_k$`, `spending_score_1_100` (gender: Male=1, Female=0).

### 4. Segmentación con KMeans

```powershell
python -m src.clustering
```

Genera:

- `reports/figures/elbow_method.png`
- `reports/figures/silhouette_scores.png`
- `reports/figures/customer_segments_pca.png`
- `reports/metrics/segment_summary.csv`
- `data/processed/mall_customers_segmented.csv`
- `models/clustering_scaler.joblib`

### 5. Levantar MLflow

```powershell
mlflow server --host 127.0.0.1 --port 5000 --backend-store-uri sqlite:///mlflow.db --default-artifact-root ./mlartifacts
```

Abre <http://127.0.0.1:5000>. Alternativa en Docker:

```powershell
docker compose --profile mlflow up -d
```

> `requirements.txt` fija `mlflow<3` y `sqlalchemy<2.1` a propósito: MLflow 3.x usa la API de *logged models* (más lenta/inestable con un servidor local) y SQLAlchemy 2.1 rompe el backend SQLite de MLflow 2.x.

### 6. Entrenar modelos

En otra terminal con el entorno activo:

```powershell
python -m src.train
```

Se entrenan `logistic_regression`, `random_forest` y `gradient_boosting`. Cada run registra accuracy, precision, recall, F1, matriz de confusión, reporte de clasificación y el modelo.

### 7. Registrar el mejor modelo

```powershell
python -m src.register_model
```

Selecciona el run con mejor F1 y le asigna el alias `champion`.

### 8. Probar la predicción local

```powershell
python -m src.predict
```

### 9. API FastAPI

```powershell
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

- Swagger UI: <http://127.0.0.1:8000/docs>
- Health: <http://127.0.0.1:8000/health>
- Métricas: <http://127.0.0.1:8000/metrics>

Ejemplo de petición:

```json
{
  "gender": "Male",
  "age": 28,
  "annual_income_k": 60,
  "spending_score": 80
}
```

```powershell
curl.exe -X POST http://127.0.0.1:8000/predict `
  -H "Content-Type: application/json" `
  -d '{\"gender\":\"Male\",\"age\":28,\"annual_income_k\":60,\"spending_score\":80}'
```

Si MLflow no está disponible, `/health` sigue respondiendo (`model_loaded: false`) y `/predict` devuelve **503** con un mensaje claro.

### 10. Frontend Streamlit

```powershell
streamlit run app/streamlit_app.py
```

Abre <http://localhost:8501>.

### 11. Monitoreo con Prometheus + Grafana

```powershell
docker compose up -d
```

- Prometheus: <http://localhost:9090> → consulta `mall_customer_prediction_count_total`
- Grafana: <http://localhost:3000> → usuario `admin`, contraseña `admin`

Grafana ya trae el datasource de Prometheus y el dashboard **Mall Customers - MLOps** aprovisionados (`docker/grafana/`). Genera tráfico desde Streamlit o con:

```powershell
1..20 | ForEach-Object {
  Invoke-RestMethod -Method Post -Uri http://127.0.0.1:8000/predict `
    -ContentType "application/json" `
    -Body '{"gender":"Female","age":35,"annual_income_k":70,"spending_score":65}'
}
```

Paneles incluidos:

- Predicciones totales
- Throughput (`rate(...[1m])`)
- Latencia p95 y promedio
- Predicciones por segmento

> **Linux**: `docker/prometheus.yml` usa `host.docker.internal`; el `extra_hosts` de `docker-compose.yml` lo resuelve también ahí.

## Tests

```powershell
pytest -q
```

Los tests de la API usan un modelo falso, así que **no necesitan MLflow** ni el modelo registrado.

## CI con GitHub Actions

El workflow `.github/workflows/ci.yml` se ejecuta en cada push a `main`/`develop` y en PRs a `main`:

1. Instala dependencias
2. Verifica que `src`, `api` y `app` compilen
3. Ejecuta preprocesamiento y clustering
4. Corre `pytest`

El dataset `data/raw/Mall_Customers.csv` está versionado a propósito: CI lo necesita.

## Subir a GitHub

```powershell
git init
git add .
git commit -m "feat: proyecto MLOps completo con MLflow, FastAPI, Streamlit y Grafana"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/mlops-mall-customers.git
git push -u origin main
```

## Notas sobre los segmentos

Los segmentos los crea KMeans y su numeración puede variar si cambias `RANDOM_STATE` o `N_CLUSTERS`. Las etiquetas comerciales del tutorial son orientativas:

| Segmento | Etiqueta sugerida |
|---|---|
| 0 | Cliente premium |
| 1 | Cliente conservador |
| 2 | Cliente joven con alto gasto |
| 3 | Cliente de bajo valor |

Revisa siempre `reports/metrics/segment_summary.csv` para asignar los nombres según los promedios reales de edad, ingreso y spending score.

## Solución de problemas

| Problema | Solución |
|---|---|
| `FileNotFoundError: Mall_Customers.csv` | Coloca el CSV en `data/raw/` con ese nombre exacto |
| `MlflowException: API request ... failed` | Levanta `mlflow server` en el puerto 5000 o revisa `MLFLOW_TRACKING_URI` |
| `ImportError: cannot import name 'FallbackAsyncAdaptedQueuePool'` | Usa `sqlalchemy<2.1` (ya está fijado en `requirements.txt`) |
| `UnicodeEncodeError: 'charmap' codec ...` al cerrar un run | `src/config.py` fuerza UTF-8; si persiste, ejecuta con `$env:PYTHONIOENCODING="utf-8"` |
| `/predict` responde 503 | El modelo no se cargó: verifica MLflow y ejecuta `python -m src.register_model` |
| Prometheus no ve la API | La API debe correr en el puerto 8000 del host y Prometheus tener `host.docker.internal` resuelto |
| `docker compose` falla con `dockerDesktopLinuxEngine` | Abre Docker Desktop y espera a que el motor esté en ejecución |
| Grafana sin datos | Espera ~10 s (scrape cada 5 s) y genera predicciones |
| Streamlit no conecta | Revisa que FastAPI esté en `http://127.0.0.1:8000` o ajusta `API_URL` |

## Nota sobre las métricas del clasificador

Los modelos alcanzan F1 ≈ 1.0 porque las etiquetas (`customer_segment`) las genera KMeans con las mismas features: el clasificador aprende una frontera que casi reproduce al clustering. Sirve para demostrar el flujo MLOps, pero **no mide rendimiento contra una verdad de negocio**. Para un caso real necesitarías etiquetas humanas (por ejemplo, segmentos definidos por marketing).

## Checklist de verificación

- [ ] `python -m src.data` genera `data/processed/mall_customers_processed.csv`
- [ ] `python -m src.clustering` genera figuras, resumen y dataset segmentado
- [ ] MLflow muestra 3 runs con métricas, matriz de confusión y modelo
- [ ] `python -m src.register_model` deja la versión 1 con alias `champion`
- [ ] `python -m src.predict` imprime un segmento
- [ ] `GET /health` devuelve `model_loaded: true`
- [ ] `POST /predict` devuelve `customer_segment`
- [ ] `GET /metrics` expone `mall_customer_prediction_count_total`
- [ ] `pytest -q` pasa
- [ ] Grafana muestra el dashboard con datos tras generar predicciones
