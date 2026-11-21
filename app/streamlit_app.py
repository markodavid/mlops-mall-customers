"""Frontend Streamlit: consume la API de FastAPI para clasificar clientes."""

import os

import requests
import streamlit as st

API_URL = os.getenv("API_URL", "http://127.0.0.1:8000")

st.set_page_config(
    page_title="Clasificador de Clientes",
    page_icon="🛍️",
    layout="centered",
)

st.title("Clasificador de Segmentos de Clientes")
st.write(
    "Esta aplicación permite probar el modelo entrenado con el dataset Mall Customers. "
    "Introduce los datos de un cliente y el sistema predice a qué segmento pertenece."
)

with st.form("customer_form"):
    gender = st.selectbox("Género", ["Male", "Female"])
    age = st.slider("Edad", min_value=18, max_value=80, value=30)
    annual_income_k = st.slider(
        "Ingreso anual en miles de dólares", min_value=10, max_value=150, value=60
    )
    spending_score = st.slider("Spending Score", min_value=1, max_value=100, value=50)

    submitted = st.form_submit_button("Predecir segmento")

if submitted:
    payload = {
        "gender": gender,
        "age": age,
        "annual_income_k": annual_income_k,
        "spending_score": spending_score,
    }

    try:
        with st.spinner("Consultando la API..."):
            response = requests.post(f"{API_URL}/predict", json=payload, timeout=10)
        result = response.json()

        if response.status_code != 200:
            st.error(result.get("detail", "Error al consultar la API"))
        else:
            segment = result["customer_segment"]
            st.success(f"Cliente clasificado en el segmento {segment}")
            st.json(result)

            st.info(
                "Recuerda: los segmentos fueron creados con clustering. "
                "Para nombrarlos comercialmente revisa "
                "reports/metrics/segment_summary.csv."
            )

    except requests.exceptions.ConnectionError:
        st.error(
            "No se pudo conectar con la API. Asegúrate de que FastAPI esté "
            f"corriendo en {API_URL}."
        )
    except requests.exceptions.Timeout:
        st.error("La API tardó demasiado en responder.")
    except ValueError:
        st.error("La API devolvió una respuesta que no es JSON válido.")
