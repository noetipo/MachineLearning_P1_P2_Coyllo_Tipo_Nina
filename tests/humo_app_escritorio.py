"""Prueba de humo de la app de escritorio. La ejecuta tests/test_apps.py en un
subproceso (así un problema de la pantalla no bloquea ni rompe a pytest)."""
import sys
import tempfile
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ / "src"), str(RAIZ / "app")]

import app_escritorio  # noqa: E402
from churn.datos import RUTA_CSV  # noqa: E402

app = app_escritorio.AppChurn()
app.withdraw()
app.update()
assert app.pestanas.index("end") == 3
r = app.estimar()
assert r is not None and 0 <= r["probabilidad"] <= 1
assert app.lbl_prob.cget("text").endswith("%")
app.limpiar()
assert app.lbl_prob.cget("text") == "—"
app.variables["servicio_internet"].set("Sin internet")
app._ajustar_dependientes()
assert app.datos_formulario()["soporte_tecnico"] == "Sin internet"
res = app.cargar_csv(str(RUTA_CSV))
assert len(res) == 7043 and len(app.tabla.get_children()) == 7043
app.ordenar("prob_abandono")
with tempfile.TemporaryDirectory() as d:
    destino = Path(d) / "salida.csv"
    app.exportar_csv(str(destino), avisar=False)
    assert destino.exists()
app.update()
app.destroy()
print("OK")
