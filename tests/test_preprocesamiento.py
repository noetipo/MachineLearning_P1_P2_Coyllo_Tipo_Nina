"""Pruebas del Pipeline de preprocesamiento."""
import numpy as np
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

from churn.datos import CATEGORICAS, NUMERICAS, separar_X_y
from churn.preprocesamiento import construir_pipeline, construir_preprocesador


def test_pipeline_ajusta_y_predice(df):
    X, y = separar_X_y(df)
    pipe = construir_pipeline(LogisticRegression(max_iter=1000)).fit(X.iloc[:2000], y.iloc[:2000])
    prob = pipe.predict_proba(X.iloc[2000:2100])[:, 1]
    assert prob.shape == (100,) and np.all((prob >= 0) & (prob <= 1))


def test_preprocesador_tolera_faltantes_y_categorias_no_vistas(df):
    X, _ = separar_X_y(df)
    prep = construir_preprocesador().fit(X)
    raro = X.iloc[:3].copy()
    raro.loc[raro.index[0], "cargo_mensual"] = np.nan
    raro.loc[raro.index[1], "tipo_contrato"] = None
    raro.loc[raro.index[2], "metodo_pago"] = "Criptomoneda"  # categoría nunca vista
    Z = prep.transform(raro)
    assert Z.shape[0] == 3 and not np.isnan(Z).any()


def test_todo_el_preprocesamiento_esta_dentro_del_pipeline():
    pipe = construir_pipeline(LogisticRegression())
    assert isinstance(pipe, Pipeline)
    assert list(pipe.named_steps) == ["prep", "modelo"]
    prep = pipe.named_steps["prep"]
    assert isinstance(prep, ColumnTransformer)
    pasos = {nombre: list(t.named_steps) for nombre, t, _ in prep.transformers}
    assert pasos == {"num": ["imputar", "escalar"], "cat": ["imputar", "onehot"]}
    cols = {nombre: c for nombre, _, c in prep.transformers}
    assert cols["num"] == NUMERICAS and cols["cat"] == CATEGORICAS


def test_escalado_aprendido_solo_con_datos_de_ajuste(df):
    """La media del escalador debe ser la del subconjunto con que se ajusta (no la global)."""
    X, _ = separar_X_y(df)
    sub = X.iloc[:500]
    prep = construir_preprocesador().fit(sub)
    escalador = prep.named_transformers_["num"].named_steps["escalar"]
    assert np.allclose(escalador.mean_, sub[NUMERICAS].mean().values)
