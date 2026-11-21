# Registro de decisiones metodológicas

Cada entrada indica la decisión adoptada y su justificación. Las cifras citadas provienen de los archivos de `resultados/`.

## Entorno y fuentes

1. **Requisitos de referencia.** Los requisitos se tomaron de las indicaciones del curso para la P1 y la P2. El repositorio de guías del docente (`ML-UNAJ-2026/GUIA-P1-P2`) no es de acceso público; conviene contrastar `REQUISITOS.md` con las rúbricas oficiales.
2. **Python 3.13.5 con tkinter (Tk 9.0).** Se requiere una distribución de Python que incluya tkinter para la aplicación de escritorio.
3. **Dataset:** descargado de la primera fuente indicada (repositorio de IBM en GitHub). Se verificó: 7 043 filas, 21 columnas, columnas clave presentes. SHA-256 guardado en `data/README.md` y comprobado en los tests.
4. **Generación de PDF.** Los informes se convierten de Markdown a HTML con la librería `markdown` y se imprimen con Google Chrome en modo *headless* (`informe/generar_pdf.py`).

## Datos

5. **`servicio_internet = "No"` se traduce como "Sin internet"** (no "No"); en las demás columnas `No internet service` → "Sin internet".
6. **`adulto_mayor` (0/1) se convierte a "Sí"/"No"** para tratar todas las binarias igual (one-hot). Las binarias se muestran como casillas en la app.
7. **`cargo_total` se excluye del modelo.** Correlación 0.9996 con `antiguedad_meses × cargo_mensual`; añadirla cambia las métricas en menos de 0.01 y en sentidos opuestos según el modelo (`resultados/02_prueba_cargo_total.csv`). Es redundante, no fuga en sentido estricto.
8. **Las 22 filas duplicadas (ignorando `id_cliente`) se conservan:** son clientes distintos con el mismo perfil, no errores de captura.
9. **Imputación dentro del Pipeline** (mediana para numéricas, moda para categóricas), nunca antes de dividir.

## Modelado

10. **Baseline = `DummyClassifier(strategy="most_frequent")`**: muestra que la exactitud de 73.5 % no sirve si el recall es 0.
11. **Modelo de la familia k-NN/SVM: SVM con kernel RBF** (`class_weight="balanced"`), sin `probability=True`; ROC-AUC y PR-AUC se calculan con `decision_function`.
12. **Desbalance: `class_weight="balanced"`** en todos los modelos que lo admiten; en la red neuronal (que no lo admite) se usan pesos por muestra balanceados, que es equivalente.
13. **Regla de elección de la P1 fijada antes de ver el test:** mayor recall medio; si hay varios a menos de una desviación del líder, se desempata por PR-AUC. Resultado: regresión logística L2.
14. **Optuna optimiza PR-AUC, no recall:** el recall se puede inflar marcando a casi todos como riesgo; PR-AUC premia ordenar bien y el recall se fija con el umbral. 40 intentos, TPE con semilla 42, CV de 5 folds solo con el entrenamiento.
15. **Regla de decisión de la P2 fijada de antemano:** el optimizado reemplaza al mejor de la P1 solo si gana en recall con p < 0.05 en Wilcoxon (25 folds) y no pierde significativamente en PR-AUC. Ganó (19 de 25 folds, p = 0.0014) → **modelo final: LightGBM optimizado**. Se declara que la mejora es pequeña (+0.013 de recall) y que el valor p es algo optimista (folds no independientes y Optuna ajustado con los mismos datos).
16. **Umbral 0.5:** con pesos balanceados ya favorece el recall; otro umbral depende del presupuesto de retención (dato que no tenemos). La tabla `resultados/03_umbral.csv` queda como guía.
17. **`n_jobs=1` en LightGBM** para que los resultados sean exactamente reproducibles.
18. **Desviaciones estándar poblacionales (ddof = 0)** en todas las tablas de validación cruzada, para que coincidan entre notebooks.

## No supervisado y red

19. **k-means con k = 3**, no k = 2. La silueta máxima es k = 2 (0.26), pero ese corte solo separa "con internet / sin internet". k = 3 es el codo de la inercia y la segunda mejor silueta (0.22), y da perfiles útiles. Agrupamiento hecho solo con el entrenamiento y sin la etiqueta.
20. **No se usó Isolation Forest** (opcional): no aportaba a la pregunta del proyecto y alargaba el análisis.
21. **Red neuronal:** `MLPClassifier(32, 16)`, *early stopping*, misma validación de 25 folds. No se adopta (empata con la regresión logística en recall y pierde con LightGBM optimizado).

## Interpretación y app

22. **Importancia por permutación con ROC-AUC** (más estable que el recall, que depende del umbral), en el test, 10 repeticiones.
23. **SHAP con `TreeExplainer`** (o `LinearExplainer` si el modelo final fuera la regresión logística), sobre las variables ya transformadas por el Pipeline.
24. **Niveles de riesgo de la app:** bajo < 0.30 ≤ medio < 0.50 ≤ alto. El corte "alto" coincide con el umbral de decisión del modelo; el corte "medio" es una elección de negocio razonable, no un resultado del análisis.
25. **La app acepta CSV en inglés (formato IBM) o en español**, y valida rangos (antigüedad 0–120 meses, cargo mensual 0–500 USD) y categorías vistas al entrenar.
26. **En la app, si el cliente no tiene internet (o teléfono), los servicios dependientes se fijan en "Sin internet" ("Sin teléfono")** para evitar combinaciones imposibles.
27. **Notebooks generados de forma reproducible:** la celda de carga de cada notebook copia literalmente las constantes y la función `traducir` de `src/churn/datos.py`; un test comprueba que ambos producen el mismo DataFrame.
28. **Los tests de notebooks se ejecutan en una copia temporal** del proyecto para no sobrescribir `resultados/` ni el modelo.
29. **La prueba de la aplicación de escritorio se ejecuta en un subproceso con tiempo límite** y se omite si tkinter no puede abrir una ventana (por ejemplo, en servidores sin entorno gráfico).
30. **Formato numérico:** punto decimal en textos, tablas y figuras, para que coincidan con lo que muestran los notebooks.
31. **Datos de prueba derivados del conjunto de test** (`data/prueba/`, generados con `scripts/generar_datos_prueba.py`). Al usar la misma partición del entrenamiento, se garantiza que el modelo no vio esos clientes, y el archivo completo reproduce exactamente las métricas de `resultados/modelo_final.json`.
32. **Comparación con el valor real en la aplicación.** Si el CSV cargado incluye `abandona` (o `Churn`), la aplicación muestra la matriz de confusión y las métricas del lote; así puede verificarse el modelo sin recurrir a los notebooks.
