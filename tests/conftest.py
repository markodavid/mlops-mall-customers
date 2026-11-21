"""Configuración de pytest.

Desactiva la carga real del modelo desde MLflow para que los tests sean
rápidos y deterministas (no dependen de un servidor MLflow en ejecución).
"""

import os

os.environ.setdefault("LOAD_MODEL_ON_STARTUP", "0")
