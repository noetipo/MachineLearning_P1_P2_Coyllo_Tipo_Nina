"""Prueba funcional de la aplicación de escritorio con los datos de data/prueba/.

Abre la aplicación tkinter, ejecuta cada escenario de prueba y registra lo que
muestra la interfaz en docs/evidencias/prueba_app_escritorio.md. En macOS
guarda además una captura de cada escenario en docs/capturas/ (requiere
pyobjc-framework-Quartz o permiso de grabación de pantalla).

Uso:
    python scripts/capturas_app.py
"""
from __future__ import annotations

import datetime
import platform
import subprocess
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

RAIZ = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(RAIZ / "src"), str(RAIZ / "app")]

import app_escritorio  # noqa: E402

CAPTURAS = RAIZ / "docs" / "capturas"
EVIDENCIAS = RAIZ / "docs" / "evidencias"
PRUEBA = RAIZ / "data" / "prueba"


def capturar_ventana_propia(ventana, ruta: Path) -> bool:
    """Captura una ventana de este mismo proceso mediante Quartz (macOS).

    Una aplicación puede capturar sus propias ventanas sin el permiso de
    grabación de pantalla. Requiere ``pyobjc-framework-Quartz``.
    """
    try:
        import Quartz
        from Foundation import NSURL
    except ImportError:
        return False
    import os
    x, ancho = ventana.winfo_rootx(), ventana.winfo_width()
    info = Quartz.CGWindowListCopyWindowInfo(Quartz.kCGWindowListOptionAll, Quartz.kCGNullWindowID)
    candidatas = [w for w in info if w["kCGWindowOwnerPID"] == os.getpid()
                  and abs(w["kCGWindowBounds"]["Width"] - ancho) <= 2
                  and abs(w["kCGWindowBounds"]["X"] - x) <= 2]
    if not candidatas:
        return False
    imagen = Quartz.CGWindowListCreateImage(
        Quartz.CGRectNull, Quartz.kCGWindowListOptionIncludingWindow,
        candidatas[0]["kCGWindowNumber"], Quartz.kCGWindowImageBoundsIgnoreFraming)
    if imagen is None:
        return False
    destino = Quartz.CGImageDestinationCreateWithURL(
        NSURL.fileURLWithPath_(str(ruta)), "public.png", 1, None)
    Quartz.CGImageDestinationAddImage(destino, imagen, None)
    return bool(Quartz.CGImageDestinationFinalize(destino))


def capturar_region(ventana, ruta: Path) -> bool:
    """Alternativa con screencapture; requiere permiso de grabación de pantalla."""
    x, y = ventana.winfo_rootx(), ventana.winfo_rooty() - 28
    ancho, alto = ventana.winfo_width(), ventana.winfo_height() + 28
    r = subprocess.run(["screencapture", "-x", "-o", f"-R{x},{y},{ancho},{alto}", str(ruta)],
                       capture_output=True, text=True)
    return r.returncode == 0


class Registro:
    def __init__(self, app):
        self.app = app
        self.filas: list[tuple[str, str, str, str]] = []
        self.capturas_ok = sys.platform == "darwin"

    def capturar(self, nombre: str, ventana=None) -> str:
        if not self.capturas_ok:
            return "—"
        w = ventana or self.app
        for _ in range(3):
            w.update()
            w.after(300)
        ruta = CAPTURAS / f"{nombre}.png"
        if not (capturar_ventana_propia(w, ruta) or capturar_region(w, ruta)):
            self.capturas_ok = False
            print("No fue posible capturar la ventana; se registra solo el texto.")
            return "—"
        return f"![{nombre}](../capturas/{nombre}.png)"

    def anotar(self, escenario: str, entrada: str, resultado: str, nombre: str, ventana=None):
        imagen = self.capturar(nombre, ventana)
        self.filas.append((escenario, entrada, resultado, imagen))
        print(f"[{len(self.filas)}] {escenario}: {resultado}")


def texto(widget) -> str:
    return widget.cget("text").replace("\n", " ")


def error_de_archivo(app, archivo: str) -> str:
    original = messagebox.showerror
    mensajes = []
    messagebox.showerror = lambda titulo, msg, **k: mensajes.append(f"{titulo}: {msg}")
    try:
        app.cargar_csv(str(PRUEBA / archivo))
    finally:
        messagebox.showerror = original
    return mensajes[0] if mensajes else "sin error"


def dialogo(app, titulo: str, mensaje: str) -> tk.Toplevel:
    dlg = tk.Toplevel(app)
    dlg.title(titulo)
    dlg.geometry(f"600x140+{app.winfo_rootx() + 220}+{app.winfo_rooty() + 240}")
    tk.Label(dlg, text=mensaje, wraplength=560, justify="left", anchor="w").pack(fill="x", padx=16, pady=24)
    dlg.attributes("-topmost", True)
    return dlg


def main():
    CAPTURAS.mkdir(parents=True, exist_ok=True)
    EVIDENCIAS.mkdir(parents=True, exist_ok=True)
    app = app_escritorio.AppChurn()
    app.geometry("1180x760+60+60")
    app.attributes("-topmost", True)
    app.lift()
    app.update()
    reg = Registro(app)
    casos = list(app.casos)

    app.pestanas.select(0)
    app.estimar()
    reg.anotar("Predicción individual con valores iniciales", "Formulario por defecto",
               f"Probabilidad {texto(app.lbl_prob)}; {texto(app.lbl_nivel)}", "01_individual_formulario")
    for i, nombre in enumerate(casos, start=2):
        app.cargar_caso(nombre)
        reg.anotar("Caso de prueba", nombre,
                   f"Probabilidad {texto(app.lbl_prob)}; {texto(app.lbl_nivel)}. {texto(app.lbl_real)}",
                   f"{i:02d}_caso")

    app.variables["servicio_internet"].set("Sin internet")
    app._ajustar_dependientes()
    reg.anotar("Coherencia del formulario", "Servicio de internet = Sin internet",
               "Servicios dependientes fijados en: " + app.variables["soporte_tecnico"].get(),
               f"{len(casos) + 2:02d}_sin_internet")
    app.limpiar()

    n = len(casos) + 3
    app.pestanas.select(1)
    for archivo in ["clientes_test.csv", "clientes_muestra.csv", "clientes_formato_ibm.csv"]:
        app.cargar_csv(str(PRUEBA / archivo))
        reg.anotar("Predicción por lote", archivo,
                   f"{texto(app.lbl_resumen)}. {texto(app.lbl_evaluacion)}", f"{n:02d}_lote")
        n += 1
    for archivo in ["error_valor_no_numerico.csv", "error_categoria_invalida.csv",
                    "error_columna_faltante.csv"]:
        msg = error_de_archivo(app, archivo)
        dlg = dialogo(app, "Archivo no válido", msg)
        reg.anotar("Validación de entradas", archivo, msg, f"{n:02d}_error", dlg)
        dlg.destroy()
        n += 1

    app.pestanas.select(2)
    reg.anotar("Información del modelo", "Pestaña «Acerca de»",
               "Muestra modelo, umbral, fecha y métricas de test", f"{n:02d}_acerca_de")
    app.destroy()

    lineas = [
        "# Prueba funcional de la aplicación de escritorio",
        "",
        f"Fecha de ejecución: {datetime.date.today().isoformat()} · Python {platform.python_version()} "
        f"· {platform.system()} {platform.release()}",
        "",
        "Generado por `scripts/capturas_app.py`, que abre la aplicación tkinter y ejecuta "
        "cada escenario con los archivos de `data/prueba/`.",
        "",
        "| # | Escenario | Entrada | Resultado mostrado en la interfaz | Captura |",
        "|---|---|---|---|---|",
    ]
    for i, (esc, ent, res, img) in enumerate(reg.filas, start=1):
        lineas.append(f"| {i} | {esc} | {ent} | {res.replace('|', '/')} | {img} |")
    (EVIDENCIAS / "prueba_app_escritorio.md").write_text("\n".join(lineas) + "\n", encoding="utf-8")
    print("Registro: docs/evidencias/prueba_app_escritorio.md")


if __name__ == "__main__":
    main()
