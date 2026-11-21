"""Cargar el modelo guardado y predecir. Sin interfaz gráfica: lo usan la app
de escritorio, la demo de Streamlit, los scripts y los tests."""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd

from .datos import CATEGORICAS, ID, NUMERICAS, RAIZ, VARIABLES_MODELO, traducir

RUTA_MODELO = RAIZ / "app" / "modelo.joblib"
RUTA_INFO = RAIZ / "resultados" / "modelo_final.json"

# Cortes de nivel de riesgo sobre la probabilidad estimada (ver DECISIONES.md).
UMBRAL_MEDIO = 0.30
UMBRAL_ALTO = 0.50  # coincide con el umbral de decisión del modelo

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
    """Entrada inválida para el modelo (mensaje pensado para el usuario final)."""


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
    """Traduce (si hace falta) y valida un DataFrame de clientes.

    Lanza ``ErrorEntrada`` con un mensaje claro si falta una columna, si un número
    no es válido o si una categoría no fue vista al entrenar.
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
    """Devuelve el lote con columnas de probabilidad y nivel, ordenado por riesgo."""
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
    return salida.sort_values("prob_abandono", ascending=False).reset_index(drop=True)
