"""Los 5 notebooks se ejecutan de principio a fin sin errores.

Se ejecutan en una copia temporal del proyecto (data/ + notebooks/) para no
sobrescribir resultados/ ni app/modelo.joblib del repositorio.
"""
import json
import shutil

import nbformat
import pytest
from nbclient import NotebookClient

NOTEBOOKS = ["01_eda.ipynb", "02_modelos.ipynb", "03_optimizacion.ipynb",
             "04_no_supervisado_y_red.ipynb", "05_interpretacion.ipynb"]


@pytest.fixture(scope="module")
def copia(raiz, tmp_path_factory):
    destino = tmp_path_factory.mktemp("proyecto")
    shutil.copytree(raiz / "data", destino / "data")
    (destino / "notebooks").mkdir()
    for n in NOTEBOOKS:
        shutil.copy(raiz / "notebooks" / n, destino / "notebooks" / n)
    return destino


@pytest.mark.lento
def test_notebooks_corren_en_orden(copia, raiz):
    for n in NOTEBOOKS:
        nb = nbformat.read(copia / "notebooks" / n, as_version=4)
        NotebookClient(nb, timeout=1800, kernel_name="python3",
                       resources={"metadata": {"path": str(copia / "notebooks")}}).execute()
    # Los resultados de la copia deben coincidir con los guardados en el repositorio.
    nuevo = json.loads((copia / "resultados" / "modelo_final.json").read_text(encoding="utf-8"))
    guardado = json.loads((raiz / "resultados" / "modelo_final.json").read_text(encoding="utf-8"))
    assert nuevo["metricas_test"] == guardado["metricas_test"]
    assert (copia / "app" / "modelo.joblib").exists()


def test_traduccion_del_notebook_igual_a_src(raiz):
    """La celda de carga de los notebooks reproduce exactamente src/churn/datos.py."""
    import pandas as pd
    from churn.datos import cargar_datos
    nb = nbformat.read(raiz / "notebooks" / "01_eda.ipynb", as_version=4)
    celda = next(c.source for c in nb.cells if c.cell_type == "code" and "def traducir" in c.source)
    ns = {"pd": pd, "np": __import__("numpy"), "EN_REPO": True,
          "RUTA_LOCAL": raiz / "data" / "Telco-Customer-Churn.csv", "URL_CSV": "", "print": lambda *a: None}
    exec("from __future__ import annotations\n" + celda, ns)
    pd.testing.assert_frame_equal(ns["df"], cargar_datos())
