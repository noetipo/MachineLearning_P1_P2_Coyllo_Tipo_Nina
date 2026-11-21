"""Ejecutar dos veces scripts/entrenar.py produce las mismas métricas."""
import json
import subprocess
import sys

import pytest


@pytest.mark.lento
def test_entrenar_dos_veces_da_lo_mismo(raiz, tmp_path):
    fichas = []
    for i in range(2):
        info = tmp_path / f"info_{i}.json"
        subprocess.run([sys.executable, str(raiz / "scripts" / "entrenar.py"),
                        "--salida-modelo", str(tmp_path / f"m_{i}.joblib"), "--salida-info", str(info)],
                       check=True, capture_output=True)
        fichas.append(json.loads(info.read_text(encoding="utf-8")))
    assert fichas[0]["metricas_test"] == fichas[1]["metricas_test"]
    assert fichas[0]["matriz_confusion_test"] == fichas[1]["matriz_confusion_test"]
    guardada = json.loads((raiz / "resultados" / "modelo_final.json").read_text(encoding="utf-8"))
    assert fichas[0]["metricas_test"] == guardada["metricas_test"]
