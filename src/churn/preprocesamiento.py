"""Preprocesamiento: todo vive dentro de un Pipeline para evitar fuga de datos.

Las medianas, modas, medias y categorías se aprenden SOLO con los datos con que
se ajusta el Pipeline (en validación cruzada, con los folds de entrenamiento).
"""
from __future__ import annotations

from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from .datos import CATEGORICAS, NUMERICAS

SEMILLA = 42


def construir_preprocesador(numericas=None, categoricas=None) -> ColumnTransformer:
    numericas = list(NUMERICAS if numericas is None else numericas)
    categoricas = list(CATEGORICAS if categoricas is None else categoricas)
    num = Pipeline([
        ("imputar", SimpleImputer(strategy="median")),
        ("escalar", StandardScaler()),
    ])
    cat = Pipeline([
        ("imputar", SimpleImputer(strategy="most_frequent")),
        ("onehot", OneHotEncoder(handle_unknown="ignore", sparse_output=False)),
    ])
    return ColumnTransformer(
        [("num", num, numericas), ("cat", cat, categoricas)],
        remainder="drop",
    )


def construir_pipeline(modelo, numericas=None, categoricas=None) -> Pipeline:
    """Pipeline completo: preprocesamiento + modelo."""
    return Pipeline([
        ("prep", construir_preprocesador(numericas, categoricas)),
        ("modelo", modelo),
    ])
