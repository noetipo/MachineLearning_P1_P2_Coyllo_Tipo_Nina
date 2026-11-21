# Decisiones tomadas

Cada decisión indica qué se decidió y por qué. Las cifras salen de `resultados/`.

## Entorno y fuentes

1. **No se pudo clonar el repositorio de referencia del docente** (`https://github.com/ML-UNAJ-2026/GUIA-P1-P2`): GitHub responde 404 sin autenticación y la organización no muestra repositorios públicos (probablemente es privado). Se siguió la estructura y los requisitos descritos en el encargo del equipo. **Pendiente:** clonarlo con una cuenta con acceso y contrastar `REQUISITOS.md` y los informes con las guías reales.
2. **Python 3.13.5 (Homebrew) en lugar de 3.11.9 (pyenv):** el Python de pyenv no tiene tkinter (`No module named '_tkinter'`); el de Homebrew sí (Tk 9.0). El entorno virtual `.venv` se creó con él.
3. **Dataset:** descargado de la primera fuente indicada (repositorio de IBM en GitHub). Se verificó: 7 043 filas, 21 columnas, columnas clave presentes. SHA-256 guardado en `data/README.md` y comprobado en los tests.
4. **No hay `pandoc`:** los PDF de los informes se generaron convirtiendo el Markdown a HTML con la librería `markdown` (instalada en `.venv`) e imprimiéndolo con Google Chrome en modo *headless* (ver `informe/README.md`).

## Datos

5. **`servicio_internet = "No"` se traduce como "Sin internet"** (no "No"), como pide el mapeo; en las demás columnas `No internet service` → "Sin internet".
6. **`adulto_mayor` (0/1) se convierte a "Sí"/"No"** para tratar todas las binarias igual (one-hot). Las binarias se muestran como casillas en la app.
7. **`cargo_total` se excluye del modelo.** Correlación 0.9996 con `antiguedad_meses × cargo_mensual`; añadirla cambia las métricas en menos de 0.01 y en sentidos opuestos según el modelo (`resultados/02_prueba_cargo_total.csv`). Es redundante, no fuga en sentido estricto.
8. **Las 22 filas duplicadas (ignorando `id_cliente`) se conservan:** son clientes distintos con el mismo perfil, no errores de captura.
9. **Imputación dentro del Pipeline** (mediana para numéricas, moda para categóricas), nunca antes de dividir.

## Modelado

10. **Baseline = `DummyClassifier(strategy="most_frequent")`**: muestra que la exactitud de 73.5 % no sirve si el recall es 0.
11. **Modelo de la familia k-NN/SVM: SVM con kernel RBF** (`class_weight="balanced"`), sin `probability=True`; ROC-AUC y PR-AUC se calculan con `decision_function`.
12. **Desbalance: `class_weight="balanced"`** en todos los modelos que lo admiten; en la red neuronal (que no lo admite) se usan pesos por muestra balanceados, que es equivalente.
13. **Regla de elección del Producto 1 fijada antes de ver el test:** mayor recall medio; si hay varios a menos de una desviación del líder, se desempata por PR-AUC. Resultado: regresión logística L2.
14. **Optuna optimiza PR-AUC, no recall:** el recall se puede inflar marcando a casi todos como riesgo; PR-AUC premia ordenar bien y el recall se fija con el umbral. 40 intentos, TPE con semilla 42, CV de 5 folds solo con el entrenamiento.
15. **Regla de decisión del Producto 2 fijada de antemano:** el optimizado reemplaza al mejor del Producto 1 solo si gana en recall con p < 0.05 en Wilcoxon (25 folds) y no pierde significativamente en PR-AUC. Ganó (19 de 25 folds, p = 0.0014) → **modelo final: LightGBM optimizado**. Se declara que la mejora es pequeña (+0.013 de recall) y que el valor p es algo optimista (folds no independientes y Optuna ajustado con los mismos datos).
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
29. **La prueba de la app de escritorio se ejecuta en un subproceso con tiempo límite** y se omite si tkinter no puede abrir una ventana (durante la sesión, después de una primera prueba exitosa, la creación de ventanas Tk se quedó bloqueada a nivel del sistema).
30. **Formato numérico:** punto decimal en textos, tablas y figuras, para que coincidan con lo que muestran los notebooks.
