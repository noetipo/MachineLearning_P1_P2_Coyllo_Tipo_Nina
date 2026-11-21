# Aplicaciones

Hay dos aplicaciones y ambas usan la misma lógica (`src/churn/prediccion.py`) y el mismo modelo (`app/modelo.joblib`, el Pipeline completo de LightGBM optimizado).

| App | Archivo | Para qué |
|---|---|---|
| **Escritorio (principal)** | `app_escritorio.py` | Predicción individual, predicción por lote desde un CSV con exportación, y ficha del modelo. tkinter + ttk. |
| **Demo web** | `app.py` | Predicción individual en el navegador (Streamlit). Es la demo que pide la guía del Producto 2. |

## Ejecutar en local

Desde la **raíz del proyecto**, con el entorno virtual creado (ver el README principal):

```bash
# App de escritorio
.venv/bin/python app/app_escritorio.py          # Windows: .venv\Scripts\python app\app_escritorio.py

# Demo en Streamlit (se abre en http://localhost:8501)
.venv/bin/streamlit run app/app.py              # Windows: .venv\Scripts\streamlit run app\app.py
```

Si `app/modelo.joblib` no existe o no carga, regénerelo con `python scripts/entrenar.py`. El modelo solo carga de forma fiable con las versiones de `requirements.txt` (scikit-learn 1.9.1 y lightgbm 4.7.0).

El manual completo de la app de escritorio está en [`docs/MANUAL_APP.md`](../docs/MANUAL_APP.md).

## Publicar la demo en Streamlit Community Cloud

1. Suba el repositorio a GitHub (público, o privado con acceso concedido a Streamlit). Deben estar incluidos `app/app.py`, `app/modelo.joblib`, `app/requirements.txt`, `src/churn/` y `resultados/modelo_final.json`.
2. Entre a <https://share.streamlit.io> con su cuenta de GitHub y pulse **Create app** → *Deploy a public app from GitHub*.
3. Elija el repositorio, la rama (`main`) y como **Main file path** escriba `app/app.py`.
4. En **Advanced settings**, elija **Python 3.13** (la versión con la que se entrenó el modelo). Streamlit instalará `app/requirements.txt`, que está junto al archivo principal.
5. Pulse **Deploy**. Cuando termine, copie la URL (`https://<nombre>.streamlit.app`) y péguela en el README principal, en lugar de `[PENDIENTE: enlace de Streamlit Community Cloud]`.

Si el despliegue falla al cargar el modelo, casi siempre es por una versión distinta de scikit-learn o lightgbm: revise el registro (*Manage app → Logs*) y que `app/requirements.txt` tenga las versiones fijadas.
