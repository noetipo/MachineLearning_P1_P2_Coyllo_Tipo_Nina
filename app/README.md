# Aplicaciones

Ambas aplicaciones comparten la lógica de predicción (`src/churn/prediccion.py`) y el modelo serializado (`app/modelo.joblib`, Pipeline completo de LightGBM optimizado).

| Aplicación | Archivo | Funcionalidad |
|---|---|---|
| Escritorio (principal) | `app_escritorio.py` | Predicción individual con casos de prueba, predicción por lote desde CSV con comparación frente al valor real y exportación, e información del modelo (tkinter) |
| Demostración web | `app.py` | Predicción individual en el navegador (Streamlit), requerida en la P2 |

## Ejecución local

Desde la raíz del proyecto, con el entorno virtual creado:

```bash
# App de escritorio
.venv/bin/python app/app_escritorio.py          # Windows: .venv\Scripts\python app\app_escritorio.py

# Demo en Streamlit (se abre en http://localhost:8501)
.venv/bin/streamlit run app/app.py              # Windows: .venv\Scripts\streamlit run app\app.py
```

Si `app/modelo.joblib` no existe o no carga, regénerelo con `python scripts/entrenar.py`. El modelo solo carga de forma fiable con las versiones de `requirements.txt` (scikit-learn 1.9.1 y lightgbm 4.7.0).

El manual de la aplicación de escritorio está en [`docs/MANUAL_APP.md`](../docs/MANUAL_APP.md) y los datos para probarla, en [`data/prueba/`](../data/prueba/README.md).

## Publicar la demo en Streamlit Community Cloud

1. Suba el repositorio a GitHub (público, o privado con acceso concedido a Streamlit). Deben estar incluidos `app/app.py`, `app/modelo.joblib`, `app/requirements.txt`, `src/churn/` y `resultados/modelo_final.json`.
2. Entre a <https://share.streamlit.io> con su cuenta de GitHub y pulse **Create app** → *Deploy a public app from GitHub*.
3. Elija el repositorio, la rama (`main`) y como **Main file path** escriba `app/app.py`.
4. En **Advanced settings**, elija **Python 3.13** (la versión con la que se entrenó el modelo). Streamlit instalará `app/requirements.txt`, que está junto al archivo principal.
5. Pulse **Deploy**. Cuando termine, copie la URL (`https://<nombre>.streamlit.app`) y agréguela al README principal.

Si el despliegue falla al cargar el modelo, casi siempre es por una versión distinta de scikit-learn o lightgbm: revise el registro (*Manage app → Logs*) y que `app/requirements.txt` tenga las versiones fijadas.
