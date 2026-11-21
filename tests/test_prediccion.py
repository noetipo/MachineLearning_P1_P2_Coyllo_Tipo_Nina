"""Pruebas de src/churn/prediccion.py (sin interfaz)."""
import pandas as pd
import pytest

from churn.datos import VARIABLES_MODELO, cargar_crudo
from churn.prediccion import (ErrorEntrada, RECOMENDACIONES, nivel_riesgo, opciones_categoricas,
                              predecir_cliente, predecir_lote)


def test_caso_individual(modelo, cliente_base):
    r = predecir_cliente(modelo, cliente_base)
    assert 0 <= r["probabilidad"] <= 1
    assert r["nivel"] in {"bajo", "medio", "alto"}
    assert r["recomendacion"] == RECOMENDACIONES[r["nivel"]]


def test_acepta_numeros_como_texto(modelo, cliente_base):
    r1 = predecir_cliente(modelo, cliente_base)
    r2 = predecir_cliente(modelo, {**cliente_base, "antiguedad_meses": "24", "cargo_mensual": "55.0"})
    assert r1["probabilidad"] == pytest.approx(r2["probabilidad"])


def test_lote_con_csv_original_en_ingles(modelo):
    crudo = cargar_crudo().head(50)
    res = predecir_lote(modelo, crudo)
    assert len(res) == 50
    assert {"id_cliente", "prob_abandono", "nivel_riesgo"} <= set(res.columns)
    assert res["prob_abandono"].is_monotonic_decreasing
    assert set(res["id_cliente"]) == set(crudo["customerID"])


def test_lote_sin_id_genera_identificadores(modelo, cliente_base):
    res = predecir_lote(modelo, pd.DataFrame([cliente_base, cliente_base]))
    assert set(res["id_cliente"]) == {"fila_1", "fila_2"}


@pytest.mark.parametrize("cambio, mensaje", [
    ({"antiguedad_meses": "abc"}, "debe ser un número"),
    ({"cargo_mensual": -5}, "debe estar entre"),
    ({"antiguedad_meses": 500}, "debe estar entre"),
    ({"tipo_contrato": "Semanal"}, "Valor no válido en 'tipo_contrato'"),
    ({"genero": ""}, "Valor no válido en 'genero'"),
])
def test_rechaza_entradas_invalidas(modelo, cliente_base, cambio, mensaje):
    with pytest.raises(ErrorEntrada, match=mensaje):
        predecir_cliente(modelo, {**cliente_base, **cambio})


def test_rechaza_columnas_faltantes(modelo, cliente_base):
    datos = pd.DataFrame([cliente_base]).drop(columns=["tipo_contrato", "metodo_pago"])
    with pytest.raises(ErrorEntrada, match="Faltan columnas obligatorias: tipo_contrato, metodo_pago"):
        predecir_lote(modelo, datos)


def test_rechaza_lote_vacio_y_tipo_incorrecto(modelo):
    with pytest.raises(ErrorEntrada):
        predecir_lote(modelo, pd.DataFrame(columns=VARIABLES_MODELO))
    with pytest.raises(ErrorEntrada):
        predecir_cliente(modelo, ["no", "es", "dict"])


def test_opciones_salen_del_modelo(modelo):
    op = opciones_categoricas(modelo)
    assert op["tipo_contrato"] == ["Dos años", "Mensual", "Un año"]
    assert "Sin internet" in op["servicio_internet"]


def test_niveles_de_riesgo():
    assert nivel_riesgo(0.1) == "bajo"
    assert nivel_riesgo(0.3) == "medio"
    assert nivel_riesgo(0.5) == "alto"
