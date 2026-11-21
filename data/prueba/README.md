# Datos de prueba

Archivos para verificar el funcionamiento del modelo y de la aplicación de escritorio. Se generan con:

```bash
python scripts/generar_datos_prueba.py
```

Todos los clientes provienen del **conjunto de prueba**: la partición 80/20 estratificada con semilla 42, la misma del entrenamiento. Ninguno fue utilizado para ajustar el modelo, lo que comprueba `tests/test_datos_prueba.py`.

| Archivo | Filas | Columnas | Uso |
|---|---|---|---|
| `clientes_test.csv` | 1 409 | Español, con `abandona` | Conjunto de prueba completo. Al cargarlo en la pestaña *Predicción por lote*, la aplicación reproduce la matriz de confusión oficial (VP = 301, FN = 73, FP = 292, VN = 743) |
| `clientes_muestra.csv` | 30 | Español, con `abandona` | Muestra estratificada (8 abandonos, 22 permanencias) para una prueba rápida |
| `clientes_formato_ibm.csv` | 15 | Inglés (formato original), con `Churn` | Verifica que la aplicación acepta el formato original de IBM y lo traduce |
| `casos_ejemplo.csv` | 8 | Español, con `abandona` y `caso` | Casos individuales disponibles en el selector *Caso de prueba* de la aplicación |
| `error_valor_no_numerico.csv` | 5 | Español | La fila 2 contiene `antiguedad_meses = "doce"` |
| `error_categoria_invalida.csv` | 5 | Español | La fila 3 contiene `tipo_contrato = "Semanal"` |
| `error_columna_faltante.csv` | 5 | Español | Falta la columna `metodo_pago` |

## Casos individuales

| Caso | Probabilidad | Nivel | Valor real | Resultado |
|---|---|---|---|---|
| 5893-PYOLZ | 8.6 % | Bajo | Permaneció | Acierto |
| 3594-UVONA | 20.1 % | Bajo | Abandonó | Falso negativo |
| 8713-IGZSO | 37.9 % | Medio | Permaneció | Acierto |
| 1087-UDSIH | 38.4 % | Medio | Abandonó | Falso negativo |
| 9450-TRJUU | 67.4 % | Alto | Permaneció | Falso positivo |
| 8650-RHRKE | 79.4 % | Alto | Abandonó | Acierto |
| 7957-RYHQD | 49.9 % | Medio | Abandonó | Falso negativo cercano al umbral |
| 8495-PRWFH | 50.2 % | Alto | Permaneció | Falso positivo cercano al umbral |

Se incluyen deliberadamente aciertos y errores para ilustrar las limitaciones del modelo con un umbral de 0.50.

## Procedimiento de prueba en la aplicación

1. Ejecutar `python app/app_escritorio.py`.
2. **Predicción individual:** elegir un caso en *Caso de prueba* y pulsar *Cargar caso*. El formulario se rellena y se muestran la probabilidad, el nivel de riesgo, el valor real y si la predicción fue correcta.
3. **Predicción por lote:** pulsar *Cargar CSV de clientes…*; el diálogo se abre en esta carpeta. Al cargar un archivo con la columna `abandona` (o `Churn`), se muestran la columna *Real*, la matriz de confusión, el recall, la precisión y la exactitud.
4. **Validación:** cargar los archivos `error_*.csv`. La aplicación debe mostrar un mensaje que identifique el problema y seguir abierta.
