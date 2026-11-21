# Notebooks

Cada notebook es autocontenido: carga los datos (desde disco o desde la URL pública), los traduce y define su preprocesamiento en las primeras celdas, de modo que puede ejecutarse en Google Colab con *Entorno de ejecución → Reiniciar y ejecutar todo*. Las tablas se guardan en `resultados/` y las figuras en `resultados/figuras/`. Los notebooks se almacenan ya ejecutados.

| Notebook | Contenido | Práctica |
|---|---|---|
| [`01_eda.ipynb`](01_eda.ipynb) | Tamaño, tipos, faltantes, proporción de abandono (26.5 %), asociación de cada variable (V de Cramér) y 5 hallazgos: contrato, antigüedad, internet y cargo, método de pago, servicios de protección. Análisis de `cargo_total`. | P1 |
| [`02_modelos.ipynb`](02_modelos.ipynb) | Métrica de éxito (recall), división 80/20 estratificada, Pipeline sin fuga, prueba de `cargo_total`, los 5 modelos obligatorios con CV de 5 folds, elección (regresión logística L2) y evaluación única en el test. | P1 |
| [`03_optimizacion.ipynb`](03_optimizacion.ipynb) | Optuna (40 intentos) sobre LightGBM, comparación fold a fold (5 × 5) con Wilcoxon contra LightGBM original y la regresión logística, análisis del umbral y decisión del modelo final (LightGBM optimizado). | P2 |
| [`04_no_supervisado_y_red.ipynb`](04_no_supervisado_y_red.ipynb) | Parte A: k-means (k = 3 por codo y silueta), perfiles con su tasa de abandono y PCA 2D. Parte B: red neuronal MLP comparada con los mismos 25 folds. | P2 |
| [`05_interpretacion.ipynb`](05_interpretacion.ipynb) | Modelo final en el test, importancia por permutación, SHAP (global y 2 casos), revisión de sesgo por género y adulto mayor, y guardado de `app/modelo.joblib` y `resultados/modelo_final.json`. | P2 |

**Orden de ejecución:** de 01 a 05. El 04 usa los parámetros de Optuna del 03 y el 05 usa la elección del 03 (`resultados/03_seleccion.json`); si esos archivos no existen (por ejemplo, en Colab sin clonar el repositorio), usan una copia de esos valores escrita en la propia celda.

**Ejecución desde la terminal:**

```bash
cd notebooks
for n in 01_eda 02_modelos 03_optimizacion 04_no_supervisado_y_red 05_interpretacion; do
  ../.venv/bin/jupyter nbconvert --to notebook --execute --inplace $n.ipynb
done
```

Tiempo aproximado: menos de dos minutos en total; el notebook 03 (Optuna) es el más extenso.
