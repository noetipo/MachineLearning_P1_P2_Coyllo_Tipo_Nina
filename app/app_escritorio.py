"""Aplicación de escritorio para la alerta temprana de abandono de clientes.

Ejecución, desde la raíz del proyecto:
    python app/app_escritorio.py

La lógica de predicción reside en src/churn/prediccion.py; este módulo
contiene únicamente la interfaz gráfica (tkinter y ttk).
"""
from __future__ import annotations

import sys
import tkinter as tk
from pathlib import Path
from tkinter import filedialog, messagebox, ttk

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import pandas as pd  # noqa: E402

from churn.prediccion import (  # noqa: E402
    ErrorEntrada, LIMITES_NUMERICOS, UMBRAL_ALTO, UMBRAL_MEDIO, cargar_info, cargar_modelo,
    comparar_con_real, opciones_categoricas, predecir_cliente, predecir_lote,
)

TITULO = "Alerta temprana de abandono de clientes"
CARPETA_PRUEBA = RAIZ / "data" / "prueba"
RUTA_CASOS = CARPETA_PRUEBA / "casos_ejemplo.csv"
COLORES_NIVEL = {"bajo": "#1f7a3d", "medio": "#b26b00", "alto": "#b3261e"}
FONDOS_NIVEL = {"bajo": "#e3f4e8", "medio": "#fff1d6", "alto": "#fde3e1"}

# (variable, etiqueta, tipo de control), agrupadas por bloque temático
GRUPOS = [
    ("Datos del cliente", [
        ("genero", "Género", "combo"),
        ("adulto_mayor", "Adulto mayor (65+)", "check"),
        ("tiene_pareja", "Tiene pareja", "check"),
        ("tiene_dependientes", "Tiene dependientes", "check"),
        ("antiguedad_meses", "Antigüedad (meses)", "entero"),
    ]),
    ("Servicios contratados", [
        ("servicio_telefonico", "Servicio telefónico", "check"),
        ("lineas_multiples", "Líneas múltiples", "combo"),
        ("servicio_internet", "Servicio de internet", "combo"),
        ("seguridad_en_linea", "Seguridad en línea", "combo"),
        ("respaldo_en_linea", "Respaldo en línea", "combo"),
        ("proteccion_dispositivo", "Protección de dispositivo", "combo"),
        ("soporte_tecnico", "Soporte técnico", "combo"),
        ("tv_streaming", "TV por streaming", "combo"),
        ("peliculas_streaming", "Películas por streaming", "combo"),
    ]),
    ("Contrato y pagos", [
        ("tipo_contrato", "Tipo de contrato", "combo"),
        ("factura_electronica", "Factura electrónica", "check"),
        ("metodo_pago", "Método de pago", "combo"),
        ("cargo_mensual", "Cargo mensual (USD)", "decimal"),
    ]),
]
DEPENDEN_DE_INTERNET = ["seguridad_en_linea", "respaldo_en_linea", "proteccion_dispositivo",
                        "soporte_tecnico", "tv_streaming", "peliculas_streaming"]
VALORES_INICIALES = {
    "genero": "Mujer", "adulto_mayor": "No", "tiene_pareja": "No", "tiene_dependientes": "No",
    "antiguedad_meses": "12", "servicio_telefonico": "Sí", "lineas_multiples": "No",
    "servicio_internet": "DSL", "seguridad_en_linea": "No", "respaldo_en_linea": "No",
    "proteccion_dispositivo": "No", "soporte_tecnico": "No", "tv_streaming": "No",
    "peliculas_streaming": "No", "tipo_contrato": "Mensual", "factura_electronica": "Sí",
    "metodo_pago": "Cheque electrónico", "cargo_mensual": "65.00",
}
COLUMNAS_TABLA = [("id_cliente", "Cliente", 110), ("prob_abandono", "Prob. abandono", 110),
                  ("nivel_riesgo", "Nivel", 70), ("tipo_contrato", "Contrato", 90),
                  ("antiguedad_meses", "Antigüedad", 80), ("cargo_mensual", "Cargo mensual", 95),
                  ("servicio_internet", "Internet", 95), ("metodo_pago", "Método de pago", 210)]
COLUMNA_REAL = ("abandona", "Real", 80)


def cargar_casos(ruta: Path = RUTA_CASOS) -> dict[str, dict]:
    """Casos de prueba (clientes reales del conjunto de test) para el formulario."""
    if not ruta.exists():
        return {}
    casos = pd.read_csv(ruta, dtype=str)
    return {fila["caso"]: fila.drop("caso").to_dict() for _, fila in casos.iterrows()}


class AppChurn(tk.Tk):
    def __init__(self, modelo=None, info: dict | None = None):
        super().__init__()
        self.title(TITULO)
        self.geometry("980x700")
        self.minsize(820, 560)
        self.modelo = modelo if modelo is not None else cargar_modelo()
        self.info = info if info is not None else cargar_info()
        self.opciones = opciones_categoricas(self.modelo)
        self.variables: dict[str, tk.Variable] = {}
        self.widgets: dict[str, tk.Widget] = {}
        self.resultado_lote: pd.DataFrame | None = None
        self.casos = cargar_casos()
        self.real_caso: str | None = None
        self.orden_desc: dict[str, bool] = {}

        estilo = ttk.Style(self)
        estilo.configure("Titulo.TLabel", font=("Helvetica", 16, "bold"))
        estilo.configure("Prob.TLabel", font=("Helvetica", 28, "bold"))
        estilo.configure("Treeview", rowheight=22)

        ttk.Label(self, text=TITULO, style="Titulo.TLabel").pack(anchor="w", padx=14, pady=(10, 4))
        self.pestanas = ttk.Notebook(self)
        self.pestanas.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        self._pestana_individual()
        self._pestana_lote()
        self._pestana_acerca()

    # Pestaña: predicción individual
    def _pestana_individual(self):
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Predicción individual")
        marco.columnconfigure(0, weight=1)
        marco.columnconfigure(1, weight=0, minsize=360)
        marco.rowconfigure(1, weight=1)

        selector = ttk.Frame(marco)
        selector.grid(row=0, column=0, columnspan=2, sticky="ew", pady=(0, 6))
        ttk.Label(selector, text="Caso de prueba:").pack(side="left")
        self.var_caso = tk.StringVar()
        self.combo_casos = ttk.Combobox(selector, textvariable=self.var_caso, values=list(self.casos),
                                        state="readonly", width=60)
        self.combo_casos.pack(side="left", padx=6)
        ttk.Button(selector, text="Cargar caso", command=self.cargar_caso).pack(side="left")
        if not self.casos:
            self.combo_casos.configure(state="disabled")

        formulario = ttk.Frame(marco)
        formulario.grid(row=1, column=0, sticky="nsew", padx=(0, 10))
        formulario.columnconfigure(0, weight=1)
        formulario.columnconfigure(1, weight=1)

        posiciones = [(0, 0), (1, 0), (0, 1)]
        for (titulo, campos), (fila, col) in zip(GRUPOS, posiciones):
            caja = ttk.LabelFrame(formulario, text=titulo, padding=8)
            caja.grid(row=fila, column=col, rowspan=2 if col == 1 else 1, sticky="nsew", padx=4, pady=4)
            caja.columnconfigure(1, weight=1)
            for i, (var, etiqueta, tipo) in enumerate(campos):
                self._crear_campo(caja, i, var, etiqueta, tipo)

        botones = ttk.Frame(formulario)
        botones.grid(row=2, column=0, columnspan=2, sticky="w", padx=4, pady=8)
        ttk.Button(botones, text="Estimar riesgo", command=self.estimar).pack(side="left", padx=(0, 8))
        ttk.Button(botones, text="Limpiar", command=self.limpiar).pack(side="left")

        panel = ttk.LabelFrame(marco, text="Resultado", padding=12)
        panel.grid(row=1, column=1, sticky="nsew")
        panel.columnconfigure(0, weight=1)
        self.lbl_titulo_prob = ttk.Label(panel, text="Probabilidad estimada de abandono", wraplength=320)
        self.lbl_titulo_prob.grid(row=0, column=0, sticky="w")
        self.lbl_prob = ttk.Label(panel, text="—", style="Prob.TLabel")
        self.lbl_prob.grid(row=1, column=0, sticky="w", pady=(2, 8))
        self.lbl_nivel = tk.Label(panel, text="Complete el formulario y pulse «Estimar riesgo».",
                                  font=("Helvetica", 13, "bold"), anchor="w", padx=8, pady=6,
                                  bg="#f1f0ec", fg="#333333")
        self.lbl_nivel.grid(row=2, column=0, sticky="ew")
        self.lbl_reco = ttk.Label(panel, text="", wraplength=320, justify="left")
        self.lbl_reco.grid(row=3, column=0, sticky="w", pady=10)
        self.lbl_real = ttk.Label(panel, text="", wraplength=320, justify="left")
        self.lbl_real.grid(row=4, column=0, sticky="w")
        self.lbl_nota = ttk.Label(panel, foreground="#888888", wraplength=320, justify="left", text=(
            f"Niveles: bajo < {UMBRAL_MEDIO:.0%} ≤ medio < {UMBRAL_ALTO:.0%} ≤ alto.\n"
            "El modelo apoya decisiones de retención; no debe usarse para negar servicios."
        ))
        self.lbl_nota.grid(row=5, column=0, sticky="sw", pady=(20, 0))
        self.limpiar()

    def _crear_campo(self, padre, fila, var, etiqueta, tipo):
        ttk.Label(padre, text=etiqueta).grid(row=fila, column=0, sticky="w", pady=3, padx=(0, 8))
        if tipo == "combo":
            v = tk.StringVar()
            w = ttk.Combobox(padre, textvariable=v, values=self.opciones[var], state="readonly", width=30)
            if var == "servicio_internet":
                w.bind("<<ComboboxSelected>>", lambda _e: self._ajustar_dependientes())
        elif tipo == "check":
            if not {"Sí", "No"} <= set(self.opciones[var]):
                raise RuntimeError(f"El modelo no tiene las opciones Sí/No para {var}")
            v = tk.StringVar()
            w = ttk.Checkbutton(padre, variable=v, onvalue="Sí", offvalue="No")
            if var == "servicio_telefonico":
                w.configure(command=self._ajustar_dependientes)
        else:
            lo, hi = LIMITES_NUMERICOS[var]
            v = tk.StringVar()
            paso = 1 if tipo == "entero" else 0.05
            w = ttk.Spinbox(padre, textvariable=v, from_=lo, to=hi, increment=paso, width=12)
        w.grid(row=fila, column=1, sticky="w" if tipo != "combo" else "ew", pady=3)
        self.variables[var] = v
        self.widgets[var] = w

    def _ajustar_dependientes(self):
        """Sin internet (o sin teléfono), los servicios dependientes quedan fijados."""
        sin_internet = self.variables["servicio_internet"].get() == "Sin internet"
        for var in DEPENDEN_DE_INTERNET:
            w = self.widgets[var]
            if sin_internet:
                self.variables[var].set("Sin internet")
                w.configure(state="disabled")
            else:
                if self.variables[var].get() == "Sin internet":
                    self.variables[var].set("No")
                w.configure(state="readonly")
        sin_telefono = self.variables["servicio_telefonico"].get() == "No"
        if sin_telefono:
            self.variables["lineas_multiples"].set("Sin teléfono")
            self.widgets["lineas_multiples"].configure(state="disabled")
        else:
            if self.variables["lineas_multiples"].get() == "Sin teléfono":
                self.variables["lineas_multiples"].set("No")
            self.widgets["lineas_multiples"].configure(state="readonly")

    def datos_formulario(self) -> dict:
        return {var: v.get() for var, v in self.variables.items()}

    def estimar(self):
        try:
            r = predecir_cliente(self.modelo, self.datos_formulario())
        except ErrorEntrada as e:
            messagebox.showerror("Dato no válido", str(e), parent=self)
            return None
        except Exception as e:
            messagebox.showerror("Error", f"No se pudo estimar el riesgo:\n{e}", parent=self)
            return None
        self.lbl_prob.configure(text=f"{r['probabilidad']:.1%}")
        self.lbl_nivel.configure(text=f"Riesgo {r['nivel'].upper()}", fg=COLORES_NIVEL[r["nivel"]],
                                 bg=FONDOS_NIVEL[r["nivel"]])
        self.lbl_reco.configure(text=r["recomendacion"])
        if self.real_caso is not None:
            real = "abandonó" if self.real_caso == "1" else "permaneció"
            pred = "abandona" if r["probabilidad"] >= UMBRAL_ALTO else "permanece"
            acierto = (self.real_caso == "1") == (r["probabilidad"] >= UMBRAL_ALTO)
            self.lbl_real.configure(text=(
                f"Valor real del caso: el cliente {real}.\n"
                f"Predicción del modelo (umbral {UMBRAL_ALTO:.2f}): {pred} → "
                f"{'acierto' if acierto else 'error'}."))
        return r

    def cargar_caso(self, nombre: str | None = None):
        """Rellena el formulario con un caso de data/prueba/casos_ejemplo.csv."""
        nombre = nombre or self.var_caso.get()
        if nombre not in self.casos:
            return None
        caso = self.casos[nombre]
        for var, v in self.variables.items():
            v.set(caso[var])
        self._ajustar_dependientes()
        for var in ("lineas_multiples", *DEPENDEN_DE_INTERNET):
            self.variables[var].set(caso[var])
        self.real_caso = caso.get("abandona")
        self.var_caso.set(nombre)
        return self.estimar()

    def limpiar(self):
        for var, valor in VALORES_INICIALES.items():
            self.variables[var].set(valor)
        self._ajustar_dependientes()
        self.real_caso = None
        self.var_caso.set("")
        self.lbl_real.configure(text="")
        self.lbl_prob.configure(text="—")
        self.lbl_nivel.configure(text="Complete el formulario y pulse «Estimar riesgo».",
                                 fg="#333333", bg="#f1f0ec")
        self.lbl_reco.configure(text="")

    # Pestaña: predicción por lote
    def _pestana_lote(self):
        marco = ttk.Frame(self.pestanas, padding=10)
        self.pestanas.add(marco, text="Predicción por lote")
        marco.columnconfigure(0, weight=1)
        marco.rowconfigure(2, weight=1)
        barra = ttk.Frame(marco)
        barra.grid(row=0, column=0, sticky="ew")
        ttk.Button(barra, text="Cargar CSV de clientes…", command=self.cargar_csv).pack(side="left")
        self.btn_exportar = ttk.Button(barra, text="Exportar resultados…", command=self.exportar_csv,
                                       state="disabled")
        self.btn_exportar.pack(side="left", padx=8)
        ttk.Label(marco, foreground="#555555", wraplength=900, justify="left", text=(
            "El CSV puede tener las columnas originales en inglés (formato IBM) o en español. "
            "Si incluye la columna abandona (o Churn), se compara la predicción con el valor real. "
            "Los archivos de prueba están en data/prueba. Un clic en un encabezado ordena la tabla.")).grid(row=1, column=0, sticky="w", pady=6)
        contenedor = ttk.Frame(marco)
        contenedor.grid(row=2, column=0, sticky="nsew")
        contenedor.columnconfigure(0, weight=1)
        contenedor.rowconfigure(0, weight=1)
        self.tabla = ttk.Treeview(contenedor, columns=[c for c, _, _ in COLUMNAS_TABLA + [COLUMNA_REAL]],
                                  show="headings")
        for col, texto, ancho in COLUMNAS_TABLA + [COLUMNA_REAL]:
            self.tabla.heading(col, text=texto, command=lambda c=col: self.ordenar(c))
            self.tabla.column(col, width=ancho, anchor="w")
        for nivel, color in FONDOS_NIVEL.items():
            self.tabla.tag_configure(nivel, background=color, foreground="#1a1a1a")
        scroll = ttk.Scrollbar(contenedor, orient="vertical", command=self.tabla.yview)
        self.tabla.configure(yscrollcommand=scroll.set)
        self.tabla.grid(row=0, column=0, sticky="nsew")
        scroll.grid(row=0, column=1, sticky="ns")
        self.lbl_resumen = ttk.Label(marco, text="Aún no se ha cargado ningún archivo.")
        self.lbl_resumen.grid(row=3, column=0, sticky="w", pady=(6, 0))
        self.lbl_evaluacion = ttk.Label(marco, text="", justify="left")
        self.lbl_evaluacion.grid(row=4, column=0, sticky="w", pady=(4, 0))

    def cargar_csv(self, ruta: str | None = None):
        ruta = ruta or filedialog.askopenfilename(
            parent=self, title="Seleccione un CSV de clientes",
            initialdir=str(CARPETA_PRUEBA if CARPETA_PRUEBA.exists() else RAIZ),
            filetypes=[("Archivos CSV", "*.csv"), ("Todos los archivos", "*.*")])
        if not ruta:
            return None
        try:
            datos = pd.read_csv(ruta)
            resultado = predecir_lote(self.modelo, datos)
        except ErrorEntrada as e:
            messagebox.showerror("Archivo no válido", str(e), parent=self)
            return None
        except Exception as e:
            messagebox.showerror("Error al leer el archivo", f"No se pudo procesar el CSV:\n{e}", parent=self)
            return None
        self.resultado_lote = resultado
        self._mostrar_tabla(resultado)
        n = len(resultado)
        conteo = resultado["nivel_riesgo"].value_counts()
        self.lbl_resumen.configure(text=(
            f"{n} clientes evaluados · alto: {conteo.get('alto', 0)} · medio: {conteo.get('medio', 0)} · "
            f"bajo: {conteo.get('bajo', 0)}  ({Path(ruta).name})"))
        ev = comparar_con_real(resultado)
        if ev is None:
            self.lbl_evaluacion.configure(text="El archivo no incluye el valor real; no se calculan métricas.")
        else:
            self.lbl_evaluacion.configure(text=(
                f"Comparación con el valor real (umbral {UMBRAL_ALTO:.2f}): "
                f"VP = {ev['vp']} · FN = {ev['fn']} · FP = {ev['fp']} · VN = {ev['vn']}\n"
                f"Recall = {ev['recall']:.3f} · Precisión = {ev['precision']:.3f} · "
                f"Exactitud = {ev['exactitud']:.3f}"))
        self.btn_exportar.configure(state="normal")
        return resultado

    def _mostrar_tabla(self, df: pd.DataFrame):
        self.tabla.delete(*self.tabla.get_children())
        for _, fila in df.iterrows():
            valores = [f"{fila[c]:.1%}" if c == "prob_abandono" else fila[c] for c, _, _ in COLUMNAS_TABLA]
            if "abandona" in df.columns and pd.notna(fila["abandona"]):
                valores.append("Abandonó" if int(fila["abandona"]) == 1 else "Permaneció")
            else:
                valores.append("—")
            self.tabla.insert("", "end", values=valores, tags=(fila["nivel_riesgo"],))

    def ordenar(self, columna: str):
        if self.resultado_lote is None:
            return
        desc = not self.orden_desc.get(columna, columna != "prob_abandono")
        self.orden_desc[columna] = desc
        self.resultado_lote = self.resultado_lote.sort_values(columna, ascending=not desc, kind="mergesort")
        self._mostrar_tabla(self.resultado_lote)

    def exportar_csv(self, ruta: str | None = None, avisar: bool = True):
        if self.resultado_lote is None:
            messagebox.showinfo("Sin resultados", "Primero cargue un CSV de clientes.", parent=self)
            return None
        ruta = ruta or filedialog.asksaveasfilename(
            parent=self, title="Guardar resultados", defaultextension=".csv",
            initialfile="riesgo_abandono.csv", filetypes=[("Archivos CSV", "*.csv")])
        if not ruta:
            return None
        try:
            self.resultado_lote.to_csv(ruta, index=False, encoding="utf-8-sig")
        except OSError as e:
            messagebox.showerror("Error al guardar", str(e), parent=self)
            return None
        if avisar:
            messagebox.showinfo("Exportado", f"Resultados guardados en:\n{ruta}", parent=self)
        return ruta

    # Pestaña: información del modelo
    def _pestana_acerca(self):
        marco = ttk.Frame(self.pestanas, padding=16)
        self.pestanas.add(marco, text="Acerca de")
        m = self.info.get("metricas_test", {})
        mc = self.info.get("matriz_confusion_test", {})
        if m:
            metricas = (f"Recall: {m['recall']:.3f}   ·   Precisión: {m['precision']:.3f}   ·   "
                        f"ROC-AUC: {m['roc_auc']:.3f}   ·   PR-AUC: {m['pr_auc']:.3f}\n"
                        f"Matriz de confusión (test, n = {self.info.get('n_test', '?')}): "
                        f"VP = {mc.get('vp')}, FN = {mc.get('fn')}, FP = {mc.get('fp')}, VN = {mc.get('vn')}")
        else:
            metricas = "No se encontró resultados/modelo_final.json."
        texto = [
            ("Proyecto", "Alerta temprana de abandono de clientes (Telco Customer Churn, IBM)"),
            ("Curso", "Machine Learning · Maestría en Ciencia de Datos e Inteligencia Artificial\n"
                      "Escuela de Posgrado, Universidad Nacional de Juliaca (EPG UNAJ)"),
            ("Docente", "Dr. Alain Paul Herrera Urtiaga"),
            ("Equipo", "Leydy Mayumy Coyllo Mamani\nNoe Wilber Tipo Mamani\nJose Alfonso Nina Condori"),
            ("Modelo", f"{self.info.get('modelo', 'desconocido')} (umbral de decisión "
                       f"{self.info.get('umbral', 0.5)}) · entrenado el {self.info.get('fecha', '?')}"),
            ("Métricas en test", metricas),
            ("Uso responsable", "Este modelo estima un riesgo para ofrecer acciones de retención. No debe "
                                "usarse para negar servicios, subir precios ni tratar peor a ningún cliente. "
                                "Se entrenó con un dataset de muestra de IBM sobre una empresa ficticia; "
                                "antes de usarlo con clientes reales habría que reentrenarlo y validarlo."),
        ]
        marco.columnconfigure(1, weight=1)
        for i, (k, v) in enumerate(texto):
            ttk.Label(marco, text=k + ":", font=("Helvetica", 11, "bold")).grid(
                row=i, column=0, sticky="nw", pady=5, padx=(0, 12))
            ttk.Label(marco, text=v, wraplength=700, justify="left").grid(row=i, column=1, sticky="w", pady=5)


def main():
    try:
        app = AppChurn()
    except FileNotFoundError as e:
        raiz = tk.Tk()
        raiz.withdraw()
        messagebox.showerror("Modelo no encontrado", str(e))
        raiz.destroy()
        return 1
    app.mainloop()
    return 0


if __name__ == "__main__":
    sys.exit(main())
