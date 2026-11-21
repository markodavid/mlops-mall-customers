"""Segmentación de clientes con KMeans + visualizaciones.

El dataset Mall_Customers no trae variable objetivo: los segmentos se crean
con clustering y luego se usan como etiquetas para entrenar un clasificador
supervisado (ver src/train.py).
"""

import matplotlib

matplotlib.use("Agg")  # necesario para CI / entornos sin pantalla

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
from sklearn.cluster import KMeans
from sklearn.decomposition import PCA
from sklearn.metrics import silhouette_score

from src.config import (
    FIGURES_DIR,
    METRICS_DIR,
    N_CLUSTERS,
    RANDOM_STATE,
    SEGMENT_SUMMARY_PATH,
    SEGMENTED_DATA_PATH,
)
from src.data import preprocess_data
from src.features import get_features, scale_features


def create_customer_segments() -> pd.DataFrame:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    METRICS_DIR.mkdir(parents=True, exist_ok=True)

    df = preprocess_data()
    X = get_features(df)
    X_scaled, scaler = scale_features(X)

    inertias = []
    silhouette_scores = []
    cluster_range = range(2, 9)

    for k in cluster_range:
        kmeans = KMeans(n_clusters=k, random_state=RANDOM_STATE, n_init=10)
        labels = kmeans.fit_predict(X_scaled)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X_scaled, labels))

    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), inertias, marker="o")
    plt.title("Método del codo para seleccionar número de clusters")
    plt.xlabel("Número de clusters")
    plt.ylabel("Inercia")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "elbow_method.png")
    plt.close()

    plt.figure(figsize=(8, 5))
    plt.plot(list(cluster_range), silhouette_scores, marker="o")
    plt.title("Silhouette score por número de clusters")
    plt.xlabel("Número de clusters")
    plt.ylabel("Silhouette score")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "silhouette_scores.png")
    plt.close()

    kmeans = KMeans(n_clusters=N_CLUSTERS, random_state=RANDOM_STATE, n_init=10)
    df["customer_segment"] = kmeans.fit_predict(X_scaled)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(X_scaled)
    df["pca_1"] = components[:, 0]
    df["pca_2"] = components[:, 1]

    plt.figure(figsize=(8, 6))
    sns.scatterplot(
        data=df,
        x="pca_1",
        y="pca_2",
        hue="customer_segment",
        palette="Set2",
    )
    plt.title("Segmentos de clientes visualizados con PCA")
    plt.tight_layout()
    plt.savefig(FIGURES_DIR / "customer_segments_pca.png")
    plt.close()

    segment_summary = (
        df.groupby("customer_segment")
        .agg(
            {
                "age": "mean",
                "annual_income_k$": "mean",
                "spending_score_1_100": "mean",
                "gender": "mean",
            }
        )
        .round(2)
        .reset_index()
    )

    segment_summary.to_csv(SEGMENT_SUMMARY_PATH, index=False)

    SEGMENTED_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(SEGMENTED_DATA_PATH, index=False)

    # Guardamos el scaler por trazabilidad (el pipeline de train escala de nuevo)
    from joblib import dump

    from src.config import MODELS_DIR

    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    dump(scaler, MODELS_DIR / "clustering_scaler.joblib")

    print("Segmentos creados correctamente.")
    print(f"Dataset segmentado guardado en: {SEGMENTED_DATA_PATH}")
    print(f"Gráficas guardadas en: {FIGURES_DIR}")
    print(segment_summary)

    return df


if __name__ == "__main__":
    create_customer_segments()
