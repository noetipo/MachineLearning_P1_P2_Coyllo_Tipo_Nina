"""Pruebas de carga, limpieza y traducción."""
import pandas as pd
import pytest

from churn.datos import (CATEGORICAS, MAPEO_COLUMNAS, MAPEO_VALORES, NUMERICAS, OBJETIVO,
                         RUTA_CSV, VARIABLES_MODELO, cargar_crudo, separar_X_y, traducir)

COLUMNAS_CLAVE = {"customerID", "tenure", "Contract", "MonthlyCharges", "TotalCharges", "Churn"}


def test_csv_carga_y_tiene_columnas_esperadas():
    assert RUTA_CSV.exists()
    crudo = cargar_crudo()
    assert crudo.shape == (7043, 21)
    assert COLUMNAS_CLAVE <= set(crudo.columns)


def test_mapeo_de_columnas_completo():
    crudo = cargar_crudo()
    assert set(crudo.columns) == set(MAPEO_COLUMNAS)
    assert set(traducir(crudo).columns) == set(MAPEO_COLUMNAS.values())


def test_cargo_total_numerico_con_blancos_como_nan(df):
    assert pd.api.types.is_float_dtype(df["cargo_total"])
    faltantes = df["cargo_total"].isna()
    assert faltantes.sum() == 11
    assert (df.loc[faltantes, "antiguedad_meses"] == 0).all()


def test_objetivo_binario(df):
    assert set(df[OBJETIVO].unique()) == {0, 1}
    assert 0.2 < df[OBJETIVO].mean() < 0.3


def test_no_quedan_categorias_sin_traducir(df):
    en_ingles = set(MAPEO_VALORES) - set(MAPEO_VALORES.values())
    for col in CATEGORICAS:
        valores = set(df[col].dropna().unique())
        assert not (valores & en_ingles), f"{col} conserva valores en inglés: {valores & en_ingles}"
    assert set(df["servicio_internet"]) == {"DSL", "Fibra óptica", "Sin internet"}
    assert set(df["tipo_contrato"]) == {"Mensual", "Un año", "Dos años"}
    assert set(df["genero"]) == {"Hombre", "Mujer"}
    assert set(df["adulto_mayor"]) == {"Sí", "No"}


def test_traducir_es_idempotente(df):
    pd.testing.assert_frame_equal(traducir(df), df)


def test_separar_excluye_id_y_cargo_total(df):
    X, y = separar_X_y(df)
    assert list(X.columns) == VARIABLES_MODELO
    assert "id_cliente" not in X and "cargo_total" not in X
    assert set(NUMERICAS) | set(CATEGORICAS) == set(VARIABLES_MODELO)
    assert len(y) == len(X) == 7043


def test_csv_original_no_modificado(raiz):
    import hashlib
    h = hashlib.sha256((raiz / "data" / "Telco-Customer-Churn.csv").read_bytes()).hexdigest()
    assert h == "16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91"
