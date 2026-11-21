# Alerta temprana de abandono de clientes · Telco Customer Churn

**Curso:** Machine Learning · Maestría en Ciencia de Datos e Inteligencia Artificial
**Institución:** Escuela de Posgrado, Universidad Nacional de Juliaca (EPG UNAJ)
**Docente:** Dr. Alain Paul Herrera Urtiaga

**Equipo:**

- Leydy Mayumy Coyllo Mamani
- Noe Wilber Tipo Mamani
- Jose Alfonso Nina Condori

**Demo web:** [PENDIENTE: enlace de Streamlit Community Cloud]

---

## El problema

Una empresa de telecomunicaciones pierde cada mes a una parte de sus clientes. Conservar a un cliente cuesta menos que conseguir uno nuevo, así que conviene detectar a tiempo a quienes están por irse.

> **Pregunta:** con los datos actuales de un cliente, ¿qué clientes cancelarán su servicio?

- **Tipo de problema:** clasificación binaria (abandona / no abandona).
- **Métrica principal: recall** de la clase "abandona". No detectar a un cliente que se va cuesta todo su ingreso futuro; una alerta innecesaria solo cuesta una oferta de retención. Reportamos también **ROC-AUC** y, por el desbalance, **PR-AUC**.

## Los datos

**Telco Customer Churn** (IBM): 7 043 clientes de una empresa ficticia, 21 columnas (demografía, antigüedad, servicios, contrato, pago, cargos). **Abandonan 1 869 (26.5 %).** Detalles, fuente, licencia y diccionario de variables en español: [`data/README.md`](data/README.md).

## Resultados principales

| Modelo | Evaluación | Recall | Precisión | ROC-AUC | PR-AUC |
|---|---|---|---|---|---|
| Baseline (clase mayoritaria) | CV 5 folds | 0.000 | 0.000 | 0.500 | 0.265 |
| Regresión logística L2 (mejor del Producto 1) | CV 5 folds | 0.793 ± 0.033 | 0.519 | 0.845 | 0.658 |
| Regresión logística L2 | **Test** | 0.773 | 0.506 | 0.839 | 0.628 |
| LightGBM optimizado con Optuna | CV 5 × 5 folds | 0.809 ± 0.028 | 0.522 | 0.850 | 0.670 |
| Red neuronal (MLP) | CV 5 × 5 folds | 0.789 ± 0.034 | 0.518 | 0.838 | 0.641 |
| **LightGBM optimizado (modelo final)** | **Test** | **0.805** | **0.508** | **0.848** | **0.667** |

*Fuentes: `resultados/02_comparacion_modelos.csv`, `02_test_mejor_modelo.json`, `03_resumen_comparacion.csv`, `04_red_vs_modelos.csv` y `modelo_final.json`.*

En el test (1 409 clientes), el modelo final **detecta 301 de los 374 clientes que abandonan**, con 292 falsas alarmas.

## Hallazgos clave

1. **El contrato mensual concentra el abandono:** 42.7 %, frente a 11.3 % (un año) y 2.8 % (dos años).
2. **Los clientes nuevos se van más:** 47.4 % en el primer año, 9.5 % después de cuatro años.
3. **Fibra óptica y cheque electrónico** marcan riesgo (41.9 % y 45.3 % de abandono); los servicios de protección (soporte técnico, seguridad en línea) se asocian a menos de la mitad de abandono.
4. **La optimización mejora, pero poco:** LightGBM optimizado supera a la regresión logística en recall en 19 de 25 folds (+0.013; Wilcoxon p = 0.0014). Mejora real pero modesta; la red neuronal no aporta.
5. **Perfil de riesgo (k-means, sin usar la etiqueta):** "nuevos con contrato mensual", 43 % de los clientes y **45.4 % de abandono**.
6. **Interpretación y sesgo:** la antigüedad y el tipo de contrato dominan las predicciones (permutación y SHAP); el género no influye. Los adultos mayores reciben más alertas (68.5 % vs 37.2 %), en parte porque abandonan más.

## Estructura del repositorio

```
├── README.md                  ← este archivo
├── REQUISITOS.md              ← lista de verificación de los requisitos
├── DECISIONES.md              ← decisiones tomadas y por qué
├── requirements.txt           ← versiones fijadas
├── data/                      ← CSV original de IBM + README (fuente y diccionario)
├── src/churn/                 ← código compartido: datos.py, preprocesamiento.py, prediccion.py
├── notebooks/                 ← 01 EDA · 02 modelos · 03 optimización · 04 no supervisado y red · 05 interpretación
├── resultados/                ← tablas (.csv/.json) y figuras (figuras/*.png) generadas por los notebooks
├── scripts/entrenar.py        ← reproduce el modelo final
├── app/                       ← app de escritorio (tkinter), demo Streamlit y modelo.joblib
├── informe/                   ← Producto1 y Producto2 (.md y .pdf)
├── docs/                      ← MANUAL_APP.md y GUIA_DE_DEFENSA.md
└── tests/                     ← batería de pruebas con pytest
```

## Cómo reproducir

### En Google Colab

1. Abra un notebook de `notebooks/` desde GitHub en Colab (*Archivo → Abrir cuaderno → GitHub*).
2. *Entorno de ejecución → Reiniciar y ejecutar todo*. Cada notebook descarga los datos de la URL pública si no los encuentra e instala lo que falte (`optuna`, `shap`, `lightgbm`).
3. Ejecute en orden: 01 → 05.

### En local

```bash
# 1. Entorno virtual (Python 3.11+ con tkinter; se usó 3.13.5)
python3 -m venv .venv
source .venv/bin/activate              # Windows: .venv\Scripts\activate
pip install -r requirements.txt

# 2. Notebooks (o ábralos con: jupyter notebook)
cd notebooks
for n in 01_eda 02_modelos 03_optimizacion 04_no_supervisado_y_red 05_interpretacion; do
  jupyter nbconvert --to notebook --execute --inplace $n.ipynb
done
cd ..

# 3. Regenerar solo el modelo final
python scripts/entrenar.py

# 4. Pruebas (todas, incluye ejecutar los notebooks; ~1.5 min)
pytest
pytest -m "not lento"                  # solo las rápidas
```

## Cómo ejecutar las aplicaciones

```bash
python app/app_escritorio.py           # app de escritorio (principal)
streamlit run app/app.py               # demo web en http://localhost:8501
```

- **App de escritorio:** predicción individual con nivel de riesgo y recomendación, predicción por lote desde un CSV (ordenable y exportable) y ficha del modelo. Manual: [`docs/MANUAL_APP.md`](docs/MANUAL_APP.md).
- **Demo Streamlit:** predicción individual en el navegador. Cómo publicarla: [`app/README.md`](app/README.md).

> **Uso responsable:** el modelo sirve para **ofrecer** retención, no para negar servicios ni subir precios.

## Informes y documentos

- [Producto 1 · Informe técnico](informe/Producto1.md) ([PDF](informe/Producto1.pdf))
- [Producto 2 · Artículo](informe/Producto2.md) ([PDF](informe/Producto2.pdf))
- [Notebooks](notebooks/README.md) · [Decisiones](DECISIONES.md) · [Requisitos](REQUISITOS.md)
- [Guía de estudio para la defensa](docs/GUIA_DE_DEFENSA.md)

## Uso de inteligencia artificial generativa

Declaramos que **el código, los análisis y los borradores de los textos de este repositorio se generaron con herramientas de inteligencia artificial generativa** a partir de las indicaciones del equipo. Todas las cifras de los textos provienen de la ejecución real de los notebooks y están guardadas en `resultados/`; las referencias bibliográficas se limitaron a fuentes identificables. **El equipo debe revisar, ejecutar y poder explicar todo el contenido antes de entregarlo**, y es responsable de él.

## Limitaciones

Dataset de muestra de IBM sobre una empresa ficticia; una foto de un solo momento; asociaciones, no causas; el modelo no conoce el valor de cada cliente ni el costo de cada acción; mejora pequeña sobre la regresión logística; más alertas para adultos mayores. Detalle en el [Producto 2, sección 6](informe/Producto2.md#6-ética-y-limitaciones).
