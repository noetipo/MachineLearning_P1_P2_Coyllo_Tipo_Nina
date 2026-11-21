"""Pruebas de los archivos de data/prueba/ y de la comparación con el valor real."""
import json

import pandas as pd
import pytest
from sklearn.model_selection import train_test_split

from churn.prediccion import ErrorEntrada, comparar_con_real, predecir_cliente, predecir_lote
from churn.preprocesamiento import SEMILLA


@pytest.fixture(scope="module")
def carpeta(raiz):
    return raiz / "data" / "prueba"


def test_conjunto_de_test_reproduce_metricas_oficiales(raiz, carpeta, modelo):
    res = predecir_lote(modelo, pd.read_csv(carpeta / "clientes_test.csv"))
    ev = comparar_con_real(res)
    ficha = json.loads((raiz / "resultados" / "modelo_final.json").read_text(encoding="utf-8"))
    assert ev["n"] == ficha["n_test"]
    assert {k: ev[k] for k in ("vp", "fn", "fp", "vn")} == ficha["matriz_confusion_test"]
    assert round(ev["recall"], 4) == ficha["metricas_test"]["recall"]
    assert round(ev["precision"], 4) == ficha["metricas_test"]["precision"]


def test_clientes_de_prueba_no_estan_en_entrenamiento(carpeta, df):
    entrenamiento, _ = train_test_split(df, test_size=0.20, stratify=df["abandona"],
                                        random_state=SEMILLA)
    test = pd.read_csv(carpeta / "clientes_test.csv")
    assert not set(test["id_cliente"]) & set(entrenamiento["id_cliente"])


@pytest.mark.parametrize("archivo", ["clientes_muestra.csv", "clientes_formato_ibm.csv"])
def test_lotes_validos_incluyen_valor_real(carpeta, modelo, archivo):
    res = predecir_lote(modelo, pd.read_csv(carpeta / archivo))
    ev = comparar_con_real(res)
    assert ev is not None and ev["n"] == len(res)


@pytest.mark.parametrize("archivo, mensaje", [
    ("error_valor_no_numerico.csv", "debe ser un número"),
    ("error_categoria_invalida.csv", "Valor no válido en 'tipo_contrato'"),
    ("error_columna_faltante.csv", "Faltan columnas obligatorias: metodo_pago"),
])
def test_archivos_con_errores_se_rechazan(carpeta, modelo, archivo, mensaje):
    with pytest.raises(ErrorEntrada, match=mensaje):
        predecir_lote(modelo, pd.read_csv(carpeta / archivo))


def test_casos_ejemplo_coinciden_con_su_descripcion(carpeta, modelo):
    casos = pd.read_csv(carpeta / "casos_ejemplo.csv")
    for _, c in casos.iterrows():
        r = predecir_cliente(modelo, c.drop(["caso", "id_cliente", "abandona"]).to_dict())
        assert c["caso"].startswith(f"Riesgo {r['nivel']}")
        acierto = (r["probabilidad"] >= 0.5) == (c["abandona"] == 1)
        assert ("acierto" in c["caso"]) == acierto


def test_lote_sin_etiqueta_no_se_evalua(modelo, cliente_base):
    assert comparar_con_real(predecir_lote(modelo, pd.DataFrame([cliente_base]))) is None
