"""Genera los archivos de data/prueba/ a partir del conjunto de test.

Se usa la misma partición 80/20 estratificada (semilla 42) del entrenamiento,
de modo que ningún cliente de estos archivos fue visto por el modelo.

Uso:
    python scripts/generar_datos_prueba.py
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import pandas as pd  # noqa: E402
from sklearn.model_selection import train_test_split  # noqa: E402

from churn.datos import ID, OBJETIVO, VARIABLES_MODELO, cargar_crudo, cargar_datos  # noqa: E402
from churn.prediccion import cargar_modelo, predecir_lote  # noqa: E402
from churn.preprocesamiento import SEMILLA  # noqa: E402

DESTINO = RAIZ / "data" / "prueba"
COLUMNAS = [ID] + VARIABLES_MODELO + [OBJETIVO]


def indices_test(df: pd.DataFrame) -> pd.Index:
    _, idx_test = train_test_split(df.index, test_size=0.20, stratify=df[OBJETIVO],
                                   random_state=SEMILLA)
    return idx_test


def elegir_casos(test: pd.DataFrame, modelo) -> pd.DataFrame:
    """Dos clientes reales por nivel de riesgo, uno que abandonó y otro que no
    cuando el nivel lo permite, más un falso negativo y un falso positivo."""
    pred = predecir_lote(modelo, test)
    filas = []
    for nivel in ["bajo", "medio", "alto"]:
        sub = pred[pred["nivel_riesgo"] == nivel]
        for real in (0, 1):
            cand = sub[sub[OBJETIVO] == real]
            if len(cand):
                filas.append(cand.iloc[len(cand) // 2])
    fn = pred[(pred["prediccion"] == 0) & (pred[OBJETIVO] == 1)]
    fp = pred[(pred["prediccion"] == 1) & (pred[OBJETIVO] == 0)]
    filas += [fn.iloc[0], fp.iloc[-1]]
    casos = pd.DataFrame(filas).drop_duplicates(ID)
    etiquetas = []
    for _, c in casos.iterrows():
        estado = "abandonó" if c[OBJETIVO] == 1 else "permaneció"
        acierto = "acierto" if c["prediccion"] == c[OBJETIVO] else "error del modelo"
        etiquetas.append(f"Riesgo {c['nivel_riesgo']} · {estado} · {acierto} ({c[ID]})")
    casos.insert(0, "caso", etiquetas)
    return casos[["caso"] + COLUMNAS].reset_index(drop=True)


def main():
    DESTINO.mkdir(parents=True, exist_ok=True)
    df = cargar_datos()
    idx = indices_test(df)
    test = df.loc[idx, COLUMNAS].reset_index(drop=True)
    modelo = cargar_modelo()

    test.to_csv(DESTINO / "clientes_test.csv", index=False, encoding="utf-8")

    _, muestra = train_test_split(test, test_size=30, stratify=test[OBJETIVO],
                                  random_state=SEMILLA)
    muestra = muestra.reset_index(drop=True)
    muestra.to_csv(DESTINO / "clientes_muestra.csv", index=False, encoding="utf-8")

    crudo = cargar_crudo()
    ids = muestra[ID].head(15)
    ibm = crudo.set_index("customerID").loc[ids].reset_index()
    ibm.to_csv(DESTINO / "clientes_formato_ibm.csv", index=False, encoding="utf-8")

    # Archivos con un defecto cada uno, para probar los mensajes de validación.
    base = muestra.head(5).drop(columns=[OBJETIVO]).astype(object)
    no_numerico = base.copy()
    no_numerico.loc[1, "antiguedad_meses"] = "doce"
    no_numerico.to_csv(DESTINO / "error_valor_no_numerico.csv", index=False, encoding="utf-8")
    categoria = base.copy()
    categoria.loc[2, "tipo_contrato"] = "Semanal"
    categoria.to_csv(DESTINO / "error_categoria_invalida.csv", index=False, encoding="utf-8")
    base.drop(columns=["metodo_pago"]).to_csv(
        DESTINO / "error_columna_faltante.csv", index=False, encoding="utf-8")

    elegir_casos(test, modelo).to_csv(DESTINO / "casos_ejemplo.csv", index=False, encoding="utf-8")

    for f in sorted(DESTINO.glob("*.csv")):
        print(f"{f.relative_to(RAIZ)}: {len(pd.read_csv(f))} filas")


if __name__ == "__main__":
    main()
