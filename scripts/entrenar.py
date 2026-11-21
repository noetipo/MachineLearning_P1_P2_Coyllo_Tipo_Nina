"""Reproduce el entrenamiento del modelo final.

Uso:
    python scripts/entrenar.py
    python scripts/entrenar.py --salida-modelo RUTA --salida-info RUTA

La configuración del modelo se lee de resultados/03_seleccion.json (notebook 03).
Se emplean la misma partición 80/20 (semilla 42) y el mismo Pipeline del
notebook 05, por lo que las métricas obtenidas son idénticas.
"""
from __future__ import annotations

import argparse
import datetime
import json
import platform
import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import joblib  # noqa: E402
import lightgbm  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import sklearn  # noqa: E402
from lightgbm import LGBMClassifier  # noqa: E402
from sklearn.linear_model import LogisticRegression  # noqa: E402
from sklearn.metrics import (  # noqa: E402
    accuracy_score, average_precision_score, confusion_matrix, f1_score,
    precision_score, recall_score, roc_auc_score,
)
from sklearn.model_selection import train_test_split  # noqa: E402

from churn.datos import VARIABLES_MODELO, cargar_datos, separar_X_y  # noqa: E402
from churn.preprocesamiento import SEMILLA, construir_pipeline  # noqa: E402

CLASES = {"LogisticRegression": LogisticRegression, "LGBMClassifier": LGBMClassifier}
RUTA_SELECCION = RAIZ / "resultados" / "03_seleccion.json"


def dividir(df):
    X, y = separar_X_y(df)
    return train_test_split(X, y, test_size=0.20, stratify=y, random_state=SEMILLA)


def entrenar(seleccion: dict | None = None):
    """Entrena el modelo final y devuelve (pipeline, ficha)."""
    if seleccion is None:
        seleccion = json.loads(RUTA_SELECCION.read_text(encoding="utf-8"))
    X_train, X_test, y_train, y_test = dividir(cargar_datos())
    modelo = construir_pipeline(CLASES[seleccion["clase"]](**seleccion["parametros"]))
    modelo.fit(X_train, y_train)
    umbral = seleccion.get("umbral", 0.5)
    prob = modelo.predict_proba(X_test)[:, 1]
    pred = (prob >= umbral).astype(int)
    mc = confusion_matrix(y_test, pred)
    metricas = {
        "recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred),
        "roc_auc": roc_auc_score(y_test, prob), "pr_auc": average_precision_score(y_test, prob),
        "f1": f1_score(y_test, pred), "accuracy": accuracy_score(y_test, pred),
    }
    ficha = {
        "modelo": seleccion["modelo"], "clase": seleccion["clase"], "parametros": seleccion["parametros"],
        "umbral": umbral, "metricas_test": {k: round(float(v), 4) for k, v in metricas.items()},
        "matriz_confusion_test": {"vn": int(mc[0, 0]), "fp": int(mc[0, 1]),
                                  "fn": int(mc[1, 0]), "vp": int(mc[1, 1])},
        "n_train": len(X_train), "n_test": len(X_test), "variables": VARIABLES_MODELO,
        "fecha": datetime.date.today().isoformat(),
        "versiones": {"python": platform.python_version(), "scikit-learn": sklearn.__version__,
                      "lightgbm": lightgbm.__version__, "pandas": pd.__version__, "numpy": np.__version__},
    }
    return modelo, ficha


def main(argv=None):
    p = argparse.ArgumentParser(description="Entrena y guarda el modelo final.")
    p.add_argument("--salida-modelo", default=str(RAIZ / "app" / "modelo.joblib"))
    p.add_argument("--salida-info", default=str(RAIZ / "resultados" / "modelo_final.json"))
    args = p.parse_args(argv)
    modelo, ficha = entrenar()
    Path(args.salida_modelo).parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(modelo, args.salida_modelo)
    Path(args.salida_info).write_text(json.dumps(ficha, ensure_ascii=False, indent=2), encoding="utf-8")
    print(f"Modelo: {ficha['modelo']}")
    print("Métricas en test:", json.dumps(ficha["metricas_test"], ensure_ascii=False))
    print(f"Guardado en {args.salida_modelo} y {args.salida_info}")


if __name__ == "__main__":
    main()
