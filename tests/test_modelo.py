"""Pruebas del modelo guardado en app/modelo.joblib."""
import json

import numpy as np
import pytest
from sklearn.metrics import average_precision_score, precision_score, recall_score, roc_auc_score

from churn.datos import cargar_datos
from churn.prediccion import RUTA_INFO, predecir_cliente


@pytest.fixture(scope="module")
def particion():
    from entrenar import dividir
    return dividir(cargar_datos())


def test_modelo_carga_y_es_pipeline_completo(modelo):
    assert list(modelo.named_steps) == ["prep", "modelo"]


def test_predict_proba_entre_0_y_1(modelo, particion):
    _, X_test, _, _ = particion
    prob = modelo.predict_proba(X_test)[:, 1]
    assert np.all((prob >= 0) & (prob <= 1))


def test_metricas_coinciden_con_modelo_final_json(modelo, particion):
    _, X_test, _, y_test = particion
    ficha = json.loads(RUTA_INFO.read_text(encoding="utf-8"))
    prob = modelo.predict_proba(X_test)[:, 1]
    pred = (prob >= ficha["umbral"]).astype(int)
    calculadas = {"recall": recall_score(y_test, pred), "precision": precision_score(y_test, pred),
                  "roc_auc": roc_auc_score(y_test, prob), "pr_auc": average_precision_score(y_test, prob)}
    for k, v in calculadas.items():
        assert round(v, 4) == pytest.approx(ficha["metricas_test"][k], abs=1e-4), k
    assert ficha["n_test"] == len(X_test)


def test_ficha_registra_versiones(modelo):
    import lightgbm, sklearn
    ficha = json.loads(RUTA_INFO.read_text(encoding="utf-8"))
    assert ficha["versiones"]["scikit-learn"] == sklearn.__version__
    assert ficha["versiones"]["lightgbm"] == lightgbm.__version__


def test_sentido_comun_contrato_mensual_vs_dos_anios(modelo, cliente_base):
    nuevo_caro = {**cliente_base, "antiguedad_meses": 1, "tipo_contrato": "Mensual", "cargo_mensual": 105.0,
                  "servicio_internet": "Fibra óptica", "seguridad_en_linea": "No", "soporte_tecnico": "No",
                  "metodo_pago": "Cheque electrónico", "factura_electronica": "Sí"}
    leal = {**cliente_base, "antiguedad_meses": 70, "tipo_contrato": "Dos años", "cargo_mensual": 60.0}
    p_nuevo = predecir_cliente(modelo, nuevo_caro)["probabilidad"]
    p_leal = predecir_cliente(modelo, leal)["probabilidad"]
    assert p_nuevo > p_leal
    assert p_nuevo > 0.5 > p_leal
