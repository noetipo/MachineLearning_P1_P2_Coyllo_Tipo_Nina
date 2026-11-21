"""Pruebas de humo de la app de escritorio (tkinter) y de la demo de Streamlit."""
import subprocess
import sys

import pytest

SONDA = "import tkinter as tk; r = tk.Tk(); r.update(); r.destroy(); print('ok')"


def hay_pantalla() -> tuple[bool, str]:
    """Comprueba en un subproceso (con tiempo límite) si tkinter puede abrir una ventana."""
    try:
        r = subprocess.run([sys.executable, "-c", SONDA], capture_output=True, text=True, timeout=20)
    except subprocess.TimeoutExpired:
        return False, "tkinter no respondió al crear una ventana (¿pantalla bloqueada o sin sesión gráfica?)"
    if r.returncode != 0 or "ok" not in r.stdout:
        return False, (r.stderr.strip().splitlines() or ["tkinter no disponible"])[-1]
    return True, ""


def test_app_escritorio_humo(raiz):
    ok, motivo = hay_pantalla()
    if not ok:
        pytest.skip(f"No hay pantalla disponible: {motivo}")
    r = subprocess.run([sys.executable, str(raiz / "tests" / "humo_app_escritorio.py")],
                       capture_output=True, text=True, timeout=180)
    assert r.returncode == 0, r.stderr[-2000:]
    assert r.stdout.strip().endswith("OK")


def test_app_escritorio_importa_sin_abrir_ventana():
    """La lógica de la app no depende de la pantalla: importar el módulo no crea ventanas."""
    import app_escritorio
    assert app_escritorio.TITULO
    assert len(app_escritorio.GRUPOS) == 3


def test_demo_streamlit_carga_y_predice(raiz):
    from streamlit.testing.v1 import AppTest
    at = AppTest.from_file(str(raiz / "app" / "app.py"), default_timeout=60).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert at.metric[0].value.endswith("%")
