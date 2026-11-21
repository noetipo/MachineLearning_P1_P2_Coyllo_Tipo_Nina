# Guía de estudio para la defensa

Cada integrante puede recibir preguntas sobre **cualquier parte** del trabajo. Esta guía resume, con nuestras cifras reales, qué se hizo, por qué y cómo explicarlo con palabras simples. Todas las cifras están en `resultados/`.

> **Importante:** el código y los textos se generaron con herramientas de inteligencia artificial generativa. Antes de la defensa, cada integrante debe abrir los notebooks, ejecutarlos y poder explicar cada paso sin leer.

---

## 1. El proyecto en un minuto

- **Problema:** predecir qué clientes de una empresa de telecomunicaciones cancelarán su servicio (clasificación binaria), para ofrecerles retención a tiempo.
- **Datos:** Telco Customer Churn de IBM, 7 043 clientes, 21 columnas, **26.5 % abandona**. Empresa ficticia.
- **Métrica principal:** recall (detectar a los que se van), más ROC-AUC y PR-AUC.
- **Producto 1:** mejor modelo = regresión logística L2 (recall test 0.773).
- **Producto 2:** LightGBM optimizado con Optuna gana a la regresión logística por poco (+0.013 de recall, p = 0.0014). **Modelo final en el test: recall 0.805, precisión 0.508, ROC-AUC 0.848, PR-AUC 0.667** (detecta 301 de 374 que se van).
- **Hallazgos:** contrato mensual (42.7 % de abandono vs 2.8 % en dos años), clientes nuevos (47.4 % en el primer año), fibra óptica (41.9 %), cheque electrónico (45.3 %), sin servicios de protección (≈ 42 %).
- **Perfil de riesgo (k-means):** "nuevos con contrato mensual", 45.4 % de abandono.
- **Interpretación:** antigüedad y tipo de contrato dominan; el género no influye.
- **Sesgo:** sin brechas relevantes por género; más alertas para adultos mayores (68.5 % vs 37.2 %).

## 2. Explicación sencilla de cada notebook

### 01 · EDA
Miramos los datos antes de modelar: tamaño (7 043 × 21), faltantes (11 blancos en `cargo_total`, todos de clientes con 0 meses), proporción de abandono (26.5 %). Medimos con la **V de Cramér** qué variables se asocian más al abandono (contrato 0.41; género 0.008) y elegimos 5 hallazgos con gráfico y cifra. Vimos que `cargo_total` es casi `antigüedad × cargo_mensual` (r = 0.9996).

### 02 · Modelos
Definimos la métrica (recall), dividimos 80/20 estratificado, armamos el **Pipeline** (imputación + escalado + one-hot) y comparamos los 5 modelos con validación cruzada de 5 folds. Probamos con y sin `cargo_total` (no cambia nada relevante → se excluye). Gana la regresión logística (recall 0.793 ± 0.033). La evaluamos **una sola vez** en el test: recall 0.773, 289 de 374 detectados, 282 falsas alarmas.

### 03 · Optimización
Optuna prueba 40 combinaciones de hiperparámetros de LightGBM y se queda con la de mayor PR-AUC en CV (0.671). Luego comparamos optimizado, original y regresión logística en los **mismos 25 folds** (5 × 5) con **Wilcoxon**. El optimizado gana al original en 25 de 25 folds y a la logística en recall en 19 de 25 (p = 0.0014). Según la regla fijada antes, el modelo final es LightGBM optimizado. Revisamos el umbral y lo dejamos en 0.5.

### 04 · No supervisado y red
**k-means** agrupa a los clientes **sin mirar si abandonaron**; después calculamos el abandono de cada grupo. Elegimos k = 3 (codo y segunda mejor silueta). Perfiles: nuevos con contrato mensual (45.4 %), leales con muchos servicios (15.6 %), sin internet (7.2 %). PCA solo para dibujarlos en 2D. La **red neuronal (MLP)** empata con la logística en recall y pierde con LightGBM optimizado.

### 05 · Interpretación
Evaluamos el modelo final en el test, medimos qué variables usa (permutación y SHAP), revisamos el sesgo por género y edad, y guardamos el Pipeline completo en `app/modelo.joblib`.

## 3. Las decisiones clave, explicadas

**¿Por qué recall?** Porque los errores no cuestan igual. Si el modelo no avisa de un cliente que se va (falso negativo), se pierde todo su ingreso futuro. Si avisa de uno que se iba a quedar (falso positivo), solo se gasta una llamada o un descuento. Preferimos detectar a la mayoría aunque haya falsas alarmas.

**¿Qué es la fuga de datos y cómo se evitó?** Es cuando información que no tendríamos al predecir (por ejemplo, datos del test) se cuela en el entrenamiento y hace que el modelo parezca mejor de lo que es. La evitamos: (1) separando el test al inicio y usándolo una sola vez; (2) metiendo todo el preprocesamiento en un Pipeline, para que medianas, medias y categorías se aprendan solo con los datos de entrenamiento de cada fold; (3) revisando variables sospechosas (`cargo_total`, que resultó redundante, no fuga).

**¿Por qué Pipeline?** Porque junta preprocesamiento y modelo en un solo objeto: en la validación cruzada se reajusta todo en cada fold (sin fuga), y al guardarlo la app recibe datos crudos y los transforma igual que en el entrenamiento.

**¿Qué hace Optuna?** Busca automáticamente buenos hiperparámetros. En lugar de probar todas las combinaciones (búsqueda en rejilla), usa los resultados de los intentos anteriores (muestreador TPE) para proponer combinaciones prometedoras. Hicimos 40 intentos, cada uno evaluado con CV de 5 folds en el entrenamiento.

**¿Qué mide Wilcoxon?** Compara dos modelos en los mismos folds: toma la diferencia de cada fold, ordena las diferencias por tamaño y mira si las positivas y negativas están equilibradas. Si p < 0.05, es poco probable que la diferencia sea por azar. No asume que las diferencias sean normales (por eso se prefiere a la prueba t con pocos folds).

**¿Cómo se eligió k?** Probamos k de 2 a 8. La silueta (qué tan bien separado está cada punto de los otros grupos, de −1 a 1) fue máxima en k = 2 (0.26), pero ese corte solo separa "con / sin internet". La inercia tiene el codo en k = 3 (cae 7 195 de 2 a 3 y solo 2 905 de 3 a 4), y k = 3 tiene la segunda mejor silueta (0.22). Elegimos k = 3 por codo e interpretabilidad.

**¿Qué dice SHAP?** Reparte la predicción de cada cliente entre sus variables: cuánto empuja cada una hacia "abandona" o hacia "se queda". Globalmente: poca antigüedad y contrato mensual empujan fuerte hacia el abandono; contrato de dos años lo frena; también suben el riesgo la fibra óptica, no tener seguridad en línea o soporte técnico, un cargo alto y el cheque electrónico.

**¿Qué dice la revisión de sesgo?** Por género, recall 0.818 (hombres) vs 0.793 (mujeres) y alertas casi iguales: sin sesgo relevante. Por edad, los adultos mayores reciben alertas en el 68.5 % de los casos vs 37.2 % del resto; en parte porque abandonan más (44.1 % vs 23.3 %), pero la brecha de alertas es mayor que la real. Recomendamos usar el modelo solo para ofrecer beneficios y monitorear por grupo.

## 4. Preguntas probables del docente y respuestas

1. **¿Por qué no usaron la exactitud (accuracy)?**
   Porque el 73.5 % de los clientes no abandona: un modelo que dice "nadie se va" tiene 73.5 % de exactitud y recall 0. Es justo lo que obtuvo nuestro baseline.

2. **¿Qué diferencia hay entre ROC-AUC y PR-AUC y por qué reportan ambas?**
   ROC-AUC mide qué tan bien el modelo ordena a los clientes por riesgo (0.5 = azar). PR-AUC resume precisión y recall en la clase positiva y es más sensible cuando esa clase es minoritaria. Nuestro modelo final: ROC-AUC 0.848, PR-AUC 0.667 (el azar en PR-AUC sería 0.265, la proporción de abandono).

3. **¿Cómo trataron el desbalance?**
   Con `class_weight="balanced"`, que da más peso a los errores en la clase "abandona" (aprox. 2.8 veces más). En la red neuronal, que no tiene ese parámetro, usamos pesos por muestra equivalentes. Además, estratificamos la división y los folds.

4. **¿Por qué excluyeron `cargo_total`? ¿Era fuga?**
   No era fuga (se conoce al predecir). Es redundante: correlación 0.9996 con antigüedad × cargo mensual. Lo probamos con y sin ella: el recall cambia menos de 0.01 y en sentidos opuestos según el modelo. Excluirla simplifica el modelo y la app.

5. **¿Qué pasa con los 11 valores faltantes?**
   Son clientes con 0 meses (aún sin facturar). Como excluimos `cargo_total`, el modelo final no los usa; en todo caso, la imputación por mediana está dentro del Pipeline.

6. **¿Por qué el baseline es "clase mayoritaria" y no "estratificado"?**
   Porque es la referencia más simple y la que mejor muestra que la exactitud engaña: 73.5 % de exactitud sin detectar a nadie.

7. **¿Por qué ganó la regresión logística en el Producto 1, si es el modelo más simple?**
   Porque aquí las relaciones son casi aditivas: contrato, antigüedad, internet y pago suman o restan riesgo de forma bastante independiente. Además, Optuna confirmó esto: los mejores árboles de LightGBM tienen profundidad 2, es decir, un modelo casi aditivo.

8. **¿Por qué Optuna optimizó PR-AUC y no recall, si su métrica es recall?**
   Porque el recall se puede inflar marcando a casi todos como riesgo. PR-AUC premia ordenar bien a los clientes; luego el recall lo fijan los pesos de clase y el umbral. De hecho, el recall subió de 0.736 a 0.809.

9. **¿La mejora del modelo optimizado es real?**
   Sí, pero pequeña. Frente a LightGBM original gana en los 25 folds. Frente a la regresión logística gana en recall en 19 de 25 folds, +0.013 de media, p = 0.0014; en el test detecta 12 clientes más (301 vs 289). Advertimos que el valor p es optimista: los folds no son independientes y Optuna se ajustó con los mismos datos.

10. **Si la mejora es tan pequeña, ¿por qué no eligieron la regresión logística?**
    Porque fijamos la regla de antemano (gana el optimizado solo si mejora el recall con p < 0.05 sin empeorar el PR-AUC) y se cumplió. Cambiar la regla después de ver el resultado sería hacer trampa. Reconocemos que la logística es una alternativa razonable y más simple.

11. **¿Por qué 5 × 5 folds y no un solo 5-fold?**
    Con 5 folds tendríamos solo 5 diferencias para la prueba, muy pocas. Repetir 5 veces con distintas particiones da 25 comparaciones y reduce la influencia de una partición afortunada.

12. **¿Por qué Wilcoxon y no una prueba t?**
    La prueba t asume que las diferencias son normales; con pocas observaciones no lo podemos garantizar. Wilcoxon solo usa los rangos de las diferencias, es más robusta.

13. **¿Por qué el umbral es 0.5? ¿No convenía bajarlo para subir el recall?**
    Bajarlo a 0.3 sube el recall a 0.911 pero alerta al 55 % de los clientes (vs 41 % con 0.5). La decisión depende de cuántos clientes puede contactar la empresa, dato que no tenemos. Con pesos balanceados, 0.5 ya favorece el recall. Dejamos la tabla (`resultados/03_umbral.csv`) para que la empresa elija.

14. **¿Cuántas veces usaron el test?**
    Para el modelo del Producto 1 una vez (notebook 02), y para el modelo final una vez (notebook 05). Ninguna decisión de modelado se tomó mirando el test: la elección se hizo con validación cruzada en el entrenamiento.

15. **¿Qué significa una precisión de 0.508?**
    Que de cada dos clientes alertados, aproximadamente uno se iba de verdad. Es el precio de un recall alto. Como la acción es ofrecer un beneficio, una falsa alarma cuesta poco.

16. **¿Por qué k-means no usa la etiqueta? ¿Para qué sirve entonces?**
    Porque es aprendizaje no supervisado: busca grupos de clientes parecidos. La etiqueta se usa solo después, para describir cada grupo. Sirve para que la empresa entienda segmentos y priorice: el perfil "nuevos con contrato mensual" (43 % de los clientes) tiene 45.4 % de abandono.

17. **La silueta es baja (0.22). ¿Los grupos sirven?**
    Indica que los grupos no están muy separados (los datos son mayormente categóricos y continuos en antigüedad). Sirven para describir y priorizar, no como segmentos rígidos. El grupo "sin internet" sí está claramente separado.

18. **¿Por qué la red neuronal no ganó?**
    Con unas 5 600 filas y 18 variables tabulares, una red no tiene ventaja: las relaciones son casi aditivas y la regresión logística o el boosting ya las capturan. La red empató con la logística en recall (p = 0.097) y perdió en ROC-AUC y PR-AUC.

19. **¿Qué diferencia hay entre importancia por permutación y SHAP?**
    La permutación mide cuánto empeora el modelo (ROC-AUC) si desordenamos una variable: es global y en términos de desempeño. SHAP explica cada predicción individual y la dirección del efecto. Ambas coinciden: antigüedad y contrato son lo más importante.

20. **¿Por qué en los gráficos SHAP aparecen valores como "−1.282 = antiguedad_meses"?**
    Porque SHAP trabaja con las variables ya transformadas por el Pipeline: las numéricas están estandarizadas (z). −1.282 equivale a una antigüedad muy baja (1 mes en ese cliente).

21. **El modelo usa `genero` y `adulto_mayor`. ¿No es discriminatorio?**
    El género no influye (importancia por permutación ≈ 0). La edad sí, y genera más alertas en adultos mayores. Como el uso es ofrecer beneficios, no negar servicios, el daño potencial es bajo; aun así proponemos monitorear y evaluar retirar `adulto_mayor` midiendo cuánto recall se pierde.

22. **¿Qué limitaciones tiene el trabajo?**
    Dataset de muestra de una empresa ficticia; una foto en un momento del tiempo; asociaciones, no causas; el modelo no conoce el valor de cada cliente ni el costo de cada acción; la mejora sobre la logística es pequeña; brecha de alertas por edad; la mitad de las alertas son falsas alarmas.

23. **¿Cómo se garantiza la reproducibilidad?**
    Semilla 42 en todo, `n_jobs=1` en LightGBM, versiones fijadas en `requirements.txt`, notebooks autocontenidos y `scripts/entrenar.py`. Los tests ejecutan los 5 notebooks desde cero en una copia limpia y verifican que las métricas coinciden, y que entrenar dos veces da lo mismo.

24. **¿Cómo funciona la aplicación?**
    La app de escritorio (tkinter) carga `app/modelo.joblib` (Pipeline completo), toma los datos del formulario o de un CSV, valida las entradas y llama a `src/churn/prediccion.py`, que devuelve la probabilidad, el nivel (bajo < 30 % ≤ medio < 50 % ≤ alto) y una recomendación. La demo en Streamlit usa la misma lógica.

25. **¿Qué haría la empresa con este modelo mañana?**
    Calcular el riesgo de todos sus clientes cada mes, contactar primero a los de riesgo alto (especialmente nuevos con contrato mensual, fibra y cheque electrónico) con ofertas concretas: pasar a contrato anual con descuento, pago automático, soporte técnico o seguridad en línea gratis unos meses. Y medir con un grupo de control si la retención funciona de verdad.

26. **¿Cómo usaron la IA generativa?**
    El código, los análisis y los borradores se generaron con herramientas de inteligencia artificial generativa a partir de nuestras indicaciones. Lo declaramos en el README y en los informes. Nuestra responsabilidad es revisar, ejecutar y entender todo, que es lo que estamos defendiendo.

## 5. Cifras para memorizar

| Qué | Valor |
|---|---|
| Clientes / abandono | 7 043 / 26.5 % (1 869) |
| Train / test | 5 634 / 1 409 |
| Contrato mensual vs dos años | 42.7 % vs 2.8 % |
| Primer año vs 49–72 meses | 47.4 % vs 9.5 % |
| Mejor P1 (reg. logística) recall CV / test | 0.793 / 0.773 |
| LightGBM optimizado recall CV (25 folds) | 0.809 |
| Optimizado vs logística (recall) | +0.013, 19 de 25 folds, p = 0.0014 |
| Modelo final en test | recall 0.805 · precisión 0.508 · ROC-AUC 0.848 · PR-AUC 0.667 |
| Matriz de confusión final | VP 301 · FN 73 · FP 292 · VN 743 |
| Perfil de riesgo | Nuevos con contrato mensual: 45.4 % |
| Alertas adultos mayores vs resto | 68.5 % vs 37.2 % |
