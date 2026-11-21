"""Carga del modelo entrenado y funciones de predicción.

Módulo independiente de la interfaz gráfica; lo utilizan la aplicación de
escritorio, la demostración en Streamlit, los scripts y las pruebas.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .datos import CATEGORICAS, ID, NUMERICAS, OBJETIVO, RAIZ, VARIABLES_MODELO, traducir

RUTA_MODELO = RAIZ / "app" / "modelo.joblib"
RUTA_INFO = RAIZ / "resultados" / "modelo_final.json"

# Cortes de nivel de riesgo sobre la probabilidad estimada.
UMBRAL_MEDIO = 0.30
UMBRAL_ALTO = 0.50  # umbral de decisión del modelo

LIMITES_NUMERICOS = {
    "antiguedad_meses": (0, 120),
    "cargo_mensual": (0.0, 500.0),
}

RECOMENDACIONES = {
    "bajo": "Riesgo bajo: mantener la relación habitual; no requiere acción de retención.",
    "medio": "Riesgo medio: hacer seguimiento y ofrecer beneficios de fidelización "
             "(por ejemplo, soporte técnico o seguridad en línea).",
    "alto": "Riesgo alto: contactar pronto con una oferta de retención "
            "(por ejemplo, pasar a contrato anual con descuento).",
}


class ErrorEntrada(ValueError):
    """Error de validación de los datos de entrada."""


def cargar_modelo(ruta: str | Path | None = None):
    ruta = Path(ruta) if ruta is not None else RUTA_MODELO
    if not ruta.exists():
        raise FileNotFoundError(
            f"No se encontró el modelo en {ruta}. Ejecute 'python scripts/entrenar.py'."
        )
    return joblib.load(ruta)


def cargar_info(ruta: str | Path | None = None) -> dict:
    ruta = Path(ruta) if ruta is not None else RUTA_INFO
    if not ruta.exists():
        return {}
    return json.loads(Path(ruta).read_text(encoding="utf-8"))


def opciones_categoricas(modelo) -> dict[str, list[str]]:
    """Categorías vistas al entrenar, leídas del OneHotEncoder del Pipeline."""
    prep = modelo.named_steps["prep"]
    _, pipe_cat, columnas = [t for t in prep.transformers_ if t[0] == "cat"][0]
    onehot = pipe_cat.named_steps["onehot"]
    return {col: [str(c) for c in cats] for col, cats in zip(columnas, onehot.categories_)}


def nivel_riesgo(prob: float) -> str:
    if prob >= UMBRAL_ALTO:
        return "alto"
    if prob >= UMBRAL_MEDIO:
        return "medio"
    return "bajo"


def validar_clientes(df: pd.DataFrame, modelo) -> pd.DataFrame:
    """Traduce y valida un DataFrame de clientes.

    Lanza ``ErrorEntrada`` si falta una columna, si un valor numérico no es
    válido o si una categoría no fue observada durante el entrenamiento.
    """
    if not isinstance(df, pd.DataFrame) or df.empty:
        raise ErrorEntrada("No hay clientes para evaluar.")
    df = traducir(df)
    faltantes = [c for c in VARIABLES_MODELO if c not in df.columns]
    if faltantes:
        raise ErrorEntrada("Faltan columnas obligatorias: " + ", ".join(faltantes))
    X = df[VARIABLES_MODELO].copy()
    for col in NUMERICAS:
        X[col] = pd.to_numeric(X[col], errors="coerce")
        malos = X[col].isna()
        if malos.any():
            filas = ", ".join(str(i + 1) for i in np.flatnonzero(malos.values)[:5])
            raise ErrorEntrada(f"'{col}' debe ser un número (filas: {filas}).")
        lo, hi = LIMITES_NUMERICOS[col]
        fuera = (X[col] < lo) | (X[col] > hi)
        if fuera.any():
            filas = ", ".join(str(i + 1) for i in np.flatnonzero(fuera.values)[:5])
            raise ErrorEntrada(f"'{col}' debe estar entre {lo} y {hi} (filas: {filas}).")
    opciones = opciones_categoricas(modelo)
    for col in CATEGORICAS:
        X[col] = X[col].astype(str).str.strip()
        invalidos = ~X[col].isin(opciones[col])
        if invalidos.any():
            valor = X.loc[invalidos, col].iloc[0]
            raise ErrorEntrada(
                f"Valor no válido en '{col}': '{valor}'. Opciones: " + ", ".join(opciones[col])
            )
    return X


def predecir_cliente(modelo, cliente: dict) -> dict:
    """Predice un cliente (diccionario variable → valor)."""
    if not isinstance(cliente, dict):
        raise ErrorEntrada("El cliente debe ser un diccionario de variables.")
    X = validar_clientes(pd.DataFrame([cliente]), modelo)
    prob = float(modelo.predict_proba(X)[0, 1])
    nivel = nivel_riesgo(prob)
    return {"probabilidad": prob, "nivel": nivel, "recomendacion": RECOMENDACIONES[nivel]}


def predecir_lote(modelo, df: pd.DataFrame) -> pd.DataFrame:
    """Predice un lote de clientes y lo ordena de mayor a menor probabilidad.

    Si el archivo incluye la variable objetivo (``abandona`` o ``Churn``), se
    conserva en la salida para poder contrastar la predicción con el valor real.
    """
    X = validar_clientes(df, modelo)
    prob = modelo.predict_proba(X)[:, 1]
    salida = pd.DataFrame(index=X.index)
    original = traducir(df)
    if ID in original.columns:
        salida[ID] = original[ID].values
    else:
        salida[ID] = [f"fila_{i + 1}" for i in range(len(X))]
    salida = pd.concat([salida, X], axis=1)
    salida["prob_abandono"] = prob.round(4)
    salida["nivel_riesgo"] = [nivel_riesgo(p) for p in prob]
    salida["prediccion"] = (prob >= UMBRAL_ALTO).astype(int)
    if OBJETIVO in original.columns:
        salida[OBJETIVO] = pd.to_numeric(original[OBJETIVO], errors="coerce").values
    return salida.sort_values("prob_abandono", ascending=False).reset_index(drop=True)


def comparar_con_real(resultado: pd.DataFrame) -> dict | None:
    """Matriz de confusión y métricas de un lote que incluye la etiqueta real.

    Devuelve ``None`` si el lote no trae la columna ``abandona``.
    """
    if OBJETIVO not in resultado.columns or resultado[OBJETIVO].isna().any():
        return None
    real = resultado[OBJETIVO].astype(int)
    pred = resultado["prediccion"].astype(int)
    vp = int(((pred == 1) & (real == 1)).sum())
    fn = int(((pred == 0) & (real == 1)).sum())
    fp = int(((pred == 1) & (real == 0)).sum())
    vn = int(((pred == 0) & (real == 0)).sum())
    n = vp + fn + fp + vn
    return {
        "n": n, "vp": vp, "fn": fn, "fp": fp, "vn": vn,
        "recall": vp / (vp + fn) if vp + fn else float("nan"),
        "precision": vp / (vp + fp) if vp + fp else float("nan"),
        "exactitud": (vp + vn) / n if n else float("nan"),
    }
