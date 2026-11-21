# Matriz de cumplimiento de requisitos

Requisitos de la Práctica 1 (P1) y la Práctica 2 (P2) del curso de Machine Learning, con la ubicación de la evidencia correspondiente. Los requisitos se tomaron de las indicaciones del curso; el repositorio de guías del docente (`ML-UNAJ-2026/GUIA-P1-P2`) no es de acceso público, por lo que debe contrastarse con las rúbricas oficiales.

Leyenda: **Sí** = cumplido · **Pendiente** = no realizado (se indica el motivo).

## Práctica 1 (P1) · Informe técnico (EDA + modelos)

| # | Requisito | Estado | Dónde |
|---|---|---|---|
| 1.1 | Problema y pregunta claros | Sí | `informe/P1_Informe_Tecnico.md` §1, `README.md` |
| 1.2 | Datos reales, fuente citada, sin datos sintéticos | Sí | `data/Telco-Customer-Churn.csv`, `data/README.md` |
| 1.3 | Tamaño, tipos, faltantes y proporción de la clase | Sí | `notebooks/01_eda.ipynb` §1–2, `resultados/01_resumen_datos.json` |
| 1.4 | De 3 a 5 hallazgos, cada uno con gráfico y cifra | Sí (5) | `01_eda.ipynb` §4, `resultados/figuras/01_h*.png` |
| 1.5 | Cierre con hallazgos numerados e implicaciones | Sí | `01_eda.ipynb` "Hallazgos principales" |
| 1.6 | Métrica de éxito justificada por el costo del error | Sí | `02_modelos.ipynb` §1, `P1_Informe_Tecnico.md` §3 |
| 1.7 | División 80/20 estratificada; test usado una sola vez | Sí | `02_modelos.ipynb` §2 y §7 |
| 1.8 | Preprocesamiento dentro de `Pipeline` + `ColumnTransformer` | Sí | `02_modelos.ipynb`, `src/churn/preprocesamiento.py` |
| 1.9 | Explicación de cómo se evita la fuga de datos | Sí | `02_modelos.ipynb` §3, `P1_Informe_Tecnico.md` §4 |
| 1.10 | Análisis de fuga de `cargo_total` con evidencia | Sí | `02_modelos.ipynb` §4, `resultados/02_prueba_cargo_total.csv` |
| 1.11 | 5 modelos obligatorios (baseline, lineal regularizado, k-NN/SVM, RF, boosting) | Sí | `02_modelos.ipynb` §5 |
| 1.12 | CV estratificada 5 folds, promedio ± desviación | Sí | `resultados/02_comparacion_modelos.csv` |
| 1.13 | Tabla comparativa con recall, ROC-AUC y PR-AUC | Sí | `02_modelos.ipynb` §5, `P1_Informe_Tecnico.md` §6.1 |
| 1.14 | Elección del mejor modelo justificada | Sí | `02_modelos.ipynb` §6 |
| 1.15 | Evaluación única en test con matriz de confusión | Sí | `02_modelos.ipynb` §7, `resultados/figuras/02_matriz_confusion_test.png` |
| 1.16 | Informe de 5–8 páginas con las secciones pedidas | Sí (≈ 2 100 palabras + 8 figuras) | `informe/P1_Informe_Tecnico.md` / `.pdf` |
| 1.17 | Conclusiones honestas | Sí | `P1_Informe_Tecnico.md` §7 |

## Práctica 2 (P2) · Artículo (optimización, no supervisado, red, interpretación, app)

| # | Requisito | Estado | Dónde |
|---|---|---|---|
| 2.1 | Optuna sobre el boosting, ≥ 30 intentos, sembrado | Sí (40, TPE semilla 42) | `03_optimizacion.ipynb` §1 |
| 2.2 | Tratamiento del desbalance | Sí | `class_weight="balanced"`; pesos por muestra en la MLP |
| 2.3 | Análisis del umbral de decisión | Sí | `03_optimizacion.ipynb` §3, `resultados/03_umbral.csv` |
| 2.4 | Comparación fold a fold 5 × 5 (`RepeatedStratifiedKFold`) optimizado vs original vs mejor P1 | Sí | `resultados/03_comparacion_folds.csv` |
| 2.5 | Prueba de Wilcoxon: folds ganados y valor p | Sí | `resultados/03_wilcoxon.csv` |
| 2.6 | Conclusión clara mejora vs empate (en caso de empate, se elige el modelo más simple) | Sí | `03_optimizacion.ipynb` §4, `resultados/03_seleccion.json` |
| 2.7 | k-means con variables escaladas, k por silueta y codo | Sí | `04_no_supervisado_y_red.ipynb` Parte A |
| 2.8 | Perfiles con nombre y tasa de abandono calculada después | Sí | `resultados/04_perfiles.csv` |
| 2.9 | El análisis aporta al problema (qué perfil concentra el riesgo y qué haría la empresa) | Sí | `04_no_supervisado_y_red.ipynb`, `P2_Articulo.md` §4.4 |
| 2.10 | Proyección PCA 2D | Sí | `resultados/figuras/04_pca_perfiles.png` |
| 2.11 | Isolation Forest (opcional) | Pendiente: No se hizo: no aportaba a la pregunta (`DECISIONES.md` #20) | — |
| 2.12 | Red neuronal (MLP) en el mismo Pipeline, misma CV, conclusión honesta | Sí | `04_no_supervisado_y_red.ipynb` Parte B |
| 2.13 | Importancia por permutación en test | Sí | `resultados/05_importancia_permutacion.csv` |
| 2.14 | SHAP global y un par de casos individuales | Sí | `resultados/figuras/05_shap_*.png` |
| 2.15 | Revisión de sesgo: recall y precisión por `genero` y `adulto_mayor`, brechas y correcciones | Sí | `resultados/05_sesgo.csv`, `05_brechas_sesgo.csv` |
| 2.16 | Modelo final con todo el entrenamiento guardado en `app/modelo.joblib` (Pipeline completo) | Sí | `05_interpretacion.ipynb` §5 |
| 2.17 | `resultados/modelo_final.json` con nombre, métricas, fecha y versiones | Sí | `resultados/modelo_final.json` |
| 2.18 | `scripts/entrenar.py` reproduce el modelo final | Sí (métricas idénticas, verificado por test) | `scripts/entrenar.py` |
| 2.19 | Demo en Streamlit o Gradio funcionando | Sí en local (probada con `streamlit.testing`) | `app/app.py` |
| 2.20 | Enlace de la demo en el README | Pendiente: Requiere publicar en Streamlit Community Cloud (lo hace el equipo) | `app/README.md` (procedimiento de publicación) |
| 2.21 | Artículo de 6–10 páginas: resumen 150–200 palabras + palabras clave, introducción, datos, método, experimentos y resultados, interpretación, ética y limitaciones, conclusiones, referencias APA 7 | Sí (resumen de 198 palabras; ≈ 3 200 palabras + 9 figuras) | `informe/P2_Articulo.md` / `.pdf` |
| 2.22 | Trabajos relacionados breves, solo con fuentes verificables | Sí | `P2_Articulo.md` §1 |
| 2.23 | Limitaciones honestas (dataset de muestra, foto en el tiempo, sesgo, lo que no puede hacer) | Sí | `P2_Articulo.md` §6 |

## Requisitos generales

| # | Requisito | Estado | Dónde |
|---|---|---|---|
| 3.1 | Estructura de carpetas pedida | Sí | ver `README.md` |
| 3.2 | Variables y categorías en español (CSV original intacto) | Sí | `src/churn/datos.py`; hash verificado en `tests/test_datos.py` |
| 3.3 | Notebooks autocontenidos, guardados ejecutados, carga local con respaldo por URL | Sí | `notebooks/` |
| 3.4 | Figuras PNG 150 dpi con títulos y ejes en español | Sí | `resultados/figuras/` (las etiquetas fijas de SHAP se tradujeron) |
| 3.5 | `random_state=42` en todo | Sí | — |
| 3.6 | `requirements.txt` con versiones fijadas | Sí | `requirements.txt`, `app/requirements.txt` |
| 3.7 | Todo número de los textos sale de una ejecución | Sí (revisado contra `resultados/`) | — |
| 3.8 | App de escritorio tkinter: 3 pestañas, validación, lógica separada | Sí | `app/app_escritorio.py`, `src/churn/prediccion.py` |
| 3.9 | Manual de la app con PyInstaller documentado | Sí | `docs/MANUAL_APP.md` |
| 3.10 | Guía de sustentación con ~25 preguntas | Sí (26) | `docs/GUIA_DE_DEFENSA.md` |
| 3.11 | Tests con pytest pasando | Sí (45 pruebas; ver `docs/evidencias/pytest.txt`) | `tests/` |
| 3.12 | Declaración del uso de herramientas de IA | Sí | `README.md` §9, informes de la P1 y la P2 |
| 3.13 | Historial de commits en español | Sí | `git log` |
| 3.14 | `_referencia/` en `.gitignore` | Sí | `.gitignore` |
| 3.15 | PDFs de los informes | Sí (Chrome headless; sin pandoc) | `informe/*.pdf` |
| 3.16 | Contrastar con las guías y rúbricas reales del docente | Pendiente: Repositorio de referencia inaccesible | — |
| 3.17 | Datos de prueba y verificación funcional de la aplicación | Sí | `data/prueba/`, `docs/evidencias/prueba_app_escritorio.md` |
