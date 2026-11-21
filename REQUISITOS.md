# Lista de verificación de requisitos

> **Advertencia importante.** Esta lista debía extraerse de `docs/Producto1_guia.md`, `docs/Producto2_guia.md` y `docs/Guia_GitHub_Colab.md` del repositorio del docente (`https://github.com/ML-UNAJ-2026/GUIA-P1-P2`), pero **ese repositorio no fue accesible** (GitHub responde 404 sin autenticación; probablemente es privado). Por eso la lista se construyó con los requisitos que el equipo transcribió en su encargo (indicaciones del curso). **Pendiente del equipo:** clonar la referencia con una cuenta con acceso, comparar con las rúbricas reales y ajustar lo que falte.

Leyenda: ✅ cumplido · ⏳ pendiente (con motivo).

## Producto 1 · Informe técnico (EDA + modelos)

| # | Requisito | Estado | Dónde |
|---|---|---|---|
| 1.1 | Problema y pregunta claros | ✅ | `informe/Producto1.md` §1, `README.md` |
| 1.2 | Datos reales, fuente citada, sin datos sintéticos | ✅ | `data/Telco-Customer-Churn.csv`, `data/README.md` |
| 1.3 | Tamaño, tipos, faltantes y proporción de la clase | ✅ | `notebooks/01_eda.ipynb` §1–2, `resultados/01_resumen_datos.json` |
| 1.4 | De 3 a 5 hallazgos, cada uno con gráfico y cifra | ✅ (5) | `01_eda.ipynb` §4, `resultados/figuras/01_h*.png` |
| 1.5 | Cierre con hallazgos numerados e implicaciones | ✅ | `01_eda.ipynb` "Hallazgos principales" |
| 1.6 | Métrica de éxito justificada por el costo del error | ✅ | `02_modelos.ipynb` §1, `Producto1.md` §3 |
| 1.7 | División 80/20 estratificada; test usado una sola vez | ✅ | `02_modelos.ipynb` §2 y §7 |
| 1.8 | Preprocesamiento dentro de `Pipeline` + `ColumnTransformer` | ✅ | `02_modelos.ipynb`, `src/churn/preprocesamiento.py` |
| 1.9 | Explicación de cómo se evita la fuga de datos | ✅ | `02_modelos.ipynb` §3, `Producto1.md` §4 |
| 1.10 | Análisis de fuga de `cargo_total` con evidencia | ✅ | `02_modelos.ipynb` §4, `resultados/02_prueba_cargo_total.csv` |
| 1.11 | 5 modelos obligatorios (baseline, lineal regularizado, k-NN/SVM, RF, boosting) | ✅ | `02_modelos.ipynb` §5 |
| 1.12 | CV estratificada 5 folds, promedio ± desviación | ✅ | `resultados/02_comparacion_modelos.csv` |
| 1.13 | Tabla comparativa con recall, ROC-AUC y PR-AUC | ✅ | `02_modelos.ipynb` §5, `Producto1.md` §6.1 |
| 1.14 | Elección del mejor modelo justificada | ✅ | `02_modelos.ipynb` §6 |
| 1.15 | Evaluación única en test con matriz de confusión | ✅ | `02_modelos.ipynb` §7, `resultados/figuras/02_matriz_confusion_test.png` |
| 1.16 | Informe de 5–8 páginas con las secciones pedidas | ✅ (≈ 2 100 palabras + 8 figuras) | `informe/Producto1.md` / `.pdf` |
| 1.17 | Conclusiones honestas | ✅ | `Producto1.md` §7 |

## Producto 2 · Artículo (optimización, no supervisado, red, interpretación, app)

| # | Requisito | Estado | Dónde |
|---|---|---|---|
| 2.1 | Optuna sobre el boosting, ≥ 30 intentos, sembrado | ✅ (40, TPE semilla 42) | `03_optimizacion.ipynb` §1 |
| 2.2 | Tratamiento del desbalance | ✅ | `class_weight="balanced"`; pesos por muestra en la MLP |
| 2.3 | Análisis del umbral de decisión | ✅ | `03_optimizacion.ipynb` §3, `resultados/03_umbral.csv` |
| 2.4 | Comparación fold a fold 5 × 5 (`RepeatedStratifiedKFold`) optimizado vs original vs mejor P1 | ✅ | `resultados/03_comparacion_folds.csv` |
| 2.5 | Prueba de Wilcoxon: folds ganados y valor p | ✅ | `resultados/03_wilcoxon.csv` |
| 2.6 | Conclusión clara mejora vs empate (empate → modelo simple) | ✅ | `03_optimizacion.ipynb` §4, `resultados/03_seleccion.json` |
| 2.7 | k-means con variables escaladas, k por silueta y codo | ✅ | `04_no_supervisado_y_red.ipynb` Parte A |
| 2.8 | Perfiles con nombre y tasa de abandono calculada después | ✅ | `resultados/04_perfiles.csv` |
| 2.9 | El análisis aporta al problema (qué perfil concentra el riesgo y qué haría la empresa) | ✅ | `04_no_supervisado_y_red.ipynb`, `Producto2.md` §4.4 |
| 2.10 | Proyección PCA 2D | ✅ | `resultados/figuras/04_pca_perfiles.png` |
| 2.11 | Isolation Forest (opcional) | ⏳ No se hizo: no aportaba a la pregunta (`DECISIONES.md` #20) | — |
| 2.12 | Red neuronal (MLP) en el mismo Pipeline, misma CV, conclusión honesta | ✅ | `04_no_supervisado_y_red.ipynb` Parte B |
| 2.13 | Importancia por permutación en test | ✅ | `resultados/05_importancia_permutacion.csv` |
| 2.14 | SHAP global y un par de casos individuales | ✅ | `resultados/figuras/05_shap_*.png` |
| 2.15 | Revisión de sesgo: recall y precisión por `genero` y `adulto_mayor`, brechas y correcciones | ✅ | `resultados/05_sesgo.csv`, `05_brechas_sesgo.csv` |
| 2.16 | Modelo final con todo el entrenamiento → `app/modelo.joblib` (Pipeline completo) | ✅ | `05_interpretacion.ipynb` §5 |
| 2.17 | `resultados/modelo_final.json` con nombre, métricas, fecha y versiones | ✅ | `resultados/modelo_final.json` |
| 2.18 | `scripts/entrenar.py` reproduce el modelo final | ✅ (métricas idénticas, verificado por test) | `scripts/entrenar.py` |
| 2.19 | Demo en Streamlit o Gradio funcionando | ✅ en local (probada con `streamlit.testing`) | `app/app.py` |
| 2.20 | Enlace de la demo en el README | ⏳ Requiere publicar en Streamlit Community Cloud (lo hace el equipo) | `README.md`: `[PENDIENTE: enlace de Streamlit Community Cloud]` |
| 2.21 | Artículo de 6–10 páginas: resumen 150–200 palabras + palabras clave, introducción, datos, método, experimentos y resultados, interpretación, ética y limitaciones, conclusiones, referencias APA 7 | ✅ (resumen de 198 palabras; ≈ 3 200 palabras + 9 figuras) | `informe/Producto2.md` / `.pdf` |
| 2.22 | Trabajos relacionados breves, solo con fuentes verificables | ✅ | `Producto2.md` §1 |
| 2.23 | Limitaciones honestas (dataset de muestra, foto en el tiempo, sesgo, lo que no puede hacer) | ✅ | `Producto2.md` §6 |

## Requisitos generales del encargo

| # | Requisito | Estado | Dónde |
|---|---|---|---|
| 3.1 | Estructura de carpetas pedida | ✅ | ver `README.md` |
| 3.2 | Variables y categorías en español (CSV original intacto) | ✅ | `src/churn/datos.py`; hash verificado en `tests/test_datos.py` |
| 3.3 | Notebooks autocontenidos, guardados ejecutados, carga local con respaldo por URL | ✅ | `notebooks/` |
| 3.4 | Figuras PNG 150 dpi con títulos y ejes en español | ✅ | `resultados/figuras/` (las etiquetas fijas de SHAP se tradujeron) |
| 3.5 | `random_state=42` en todo | ✅ | — |
| 3.6 | `requirements.txt` con versiones fijadas | ✅ | `requirements.txt`, `app/requirements.txt` |
| 3.7 | Todo número de los textos sale de una ejecución | ✅ (revisado contra `resultados/`) | — |
| 3.8 | App de escritorio tkinter: 3 pestañas, validación, lógica separada | ✅ | `app/app_escritorio.py`, `src/churn/prediccion.py` |
| 3.9 | Manual de la app con PyInstaller documentado | ✅ | `docs/MANUAL_APP.md` |
| 3.10 | Guía de defensa con ~25 preguntas | ✅ (26) | `docs/GUIA_DE_DEFENSA.md` |
| 3.11 | Tests con pytest pasando | ✅ 35 pasan, 1 omitido (ventana Tk sin pantalla) | `tests/` |
| 3.12 | Declaración veraz del uso de IA | ✅ | `README.md`, `informe/Producto1.md`, `informe/Producto2.md` |
| 3.13 | Commits pequeños en español, sin push | ✅ | `git log` |
| 3.14 | `_referencia/` en `.gitignore` | ✅ (la carpeta no llegó a crearse) | `.gitignore` |
| 3.15 | PDFs de los informes | ✅ (Chrome headless; sin pandoc) | `informe/*.pdf` |
| 3.16 | Contrastar con las guías y rúbricas reales del docente | ⏳ Repositorio de referencia inaccesible | — |
