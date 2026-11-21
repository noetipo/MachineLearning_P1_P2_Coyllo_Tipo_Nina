"""Demostración web (Streamlit) de la alerta temprana de abandono de clientes.

Ejecución, desde la raíz del proyecto:
    streamlit run app/app.py

Comparte la lógica de predicción con la aplicación de escritorio
(src/churn/prediccion.py).
"""
from __future__ import annotations

import sys
from pathlib import Path

RAIZ = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(RAIZ / "src"))

import streamlit as st  # noqa: E402

from churn.prediccion import (  # noqa: E402
    ErrorEntrada, cargar_info, cargar_modelo, opciones_categoricas, predecir_cliente,
)

st.set_page_config(page_title="Alerta de abandono de clientes", layout="wide")


@st.cache_resource
def recursos():
    modelo = cargar_modelo(RAIZ / "app" / "modelo.joblib")
    return modelo, opciones_categoricas(modelo), cargar_info()


modelo, opciones, info = recursos()

st.title("Alerta temprana de abandono de clientes")
st.caption("Machine Learning · Maestría en Ciencia de Datos e IA · EPG UNAJ — "
           "Leydy Mayumy Coyllo Mamani · Noe Wilber Tipo Mamani · Jose Alfonso Nina Condori")


def si_no(etiqueta, clave, defecto=False):
    return "Sí" if st.checkbox(etiqueta, value=defecto, key=clave) else "No"


def lista(etiqueta, clave, defecto=None, deshabilitar=False, forzar=None):
    valores = opciones[clave]
    if forzar is not None:
        return st.selectbox(etiqueta, [forzar], key=f"{clave}_{forzar}", disabled=True)
    indice = valores.index(defecto) if defecto in valores else 0
    return st.selectbox(etiqueta, valores, index=indice, key=clave, disabled=deshabilitar)


c1, c2, c3 = st.columns(3)
with c1:
    st.subheader("Datos del cliente")
    genero = lista("Género", "genero")
    adulto_mayor = si_no("Adulto mayor (65+)", "adulto_mayor")
    tiene_pareja = si_no("Tiene pareja", "tiene_pareja")
    tiene_dependientes = si_no("Tiene dependientes", "tiene_dependientes")
    antiguedad = st.slider("Antigüedad (meses)", 0, 72, 12)
with c2:
    st.subheader("Servicios")
    telefono = si_no("Servicio telefónico", "servicio_telefonico", True)
    lineas = lista("Líneas múltiples", "lineas_multiples", "No",
                   forzar="Sin teléfono" if telefono == "No" else None)
    internet = lista("Servicio de internet", "servicio_internet", "Fibra óptica")
    forzar = "Sin internet" if internet == "Sin internet" else None
    servicios = {}
    for clave, etiqueta in [("seguridad_en_linea", "Seguridad en línea"), ("respaldo_en_linea", "Respaldo en línea"),
                            ("proteccion_dispositivo", "Protección de dispositivo"),
                            ("soporte_tecnico", "Soporte técnico"), ("tv_streaming", "TV por streaming"),
                            ("peliculas_streaming", "Películas por streaming")]:
        servicios[clave] = lista(etiqueta, clave, "No", forzar=forzar)
with c3:
    st.subheader("Contrato y pagos")
    contrato = lista("Tipo de contrato", "tipo_contrato", "Mensual")
    factura = si_no("Factura electrónica", "factura_electronica", True)
    pago = lista("Método de pago", "metodo_pago", "Cheque electrónico")
    cargo = st.number_input("Cargo mensual (USD)", min_value=0.0, max_value=500.0, value=70.0, step=0.5)

cliente = {"genero": genero, "adulto_mayor": adulto_mayor, "tiene_pareja": tiene_pareja,
           "tiene_dependientes": tiene_dependientes, "antiguedad_meses": antiguedad,
           "servicio_telefonico": telefono, "lineas_multiples": lineas, "servicio_internet": internet,
           **servicios, "tipo_contrato": contrato, "factura_electronica": factura, "metodo_pago": pago,
           "cargo_mensual": cargo}

if st.button("Estimar riesgo", type="primary"):
    try:
        r = predecir_cliente(modelo, cliente)
    except ErrorEntrada as e:
        st.error(str(e))
    else:
        st.metric("Probabilidad de abandono", f"{r['probabilidad']:.1%}")
        mensaje = {"bajo": st.success, "medio": st.warning, "alto": st.error}[r["nivel"]]
        mensaje(f"Riesgo {r['nivel'].upper()} — {r['recomendacion']}")

with st.expander("Acerca del modelo"):
    m = info.get("metricas_test", {})
    st.write(f"**Modelo:** {info.get('modelo', '?')} · **Fecha:** {info.get('fecha', '?')}")
    if m:
        st.write(f"Métricas en test: recall {m['recall']:.3f} · precisión {m['precision']:.3f} · "
                 f"ROC-AUC {m['roc_auc']:.3f} · PR-AUC {m['pr_auc']:.3f}")
    st.info("Uso responsable: el modelo sirve para ofrecer retención, no para negar servicios. "
            "Se entrenó con un dataset de muestra de IBM (empresa ficticia).")
