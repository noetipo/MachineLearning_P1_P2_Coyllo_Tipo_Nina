# Datos · Telco Customer Churn

## Fuente

- **Dataset:** *Telco Customer Churn*, dataset de muestra publicado por IBM sobre una empresa de telecomunicaciones **ficticia**.
- **Archivo:** `Telco-Customer-Churn.csv` (970 457 bytes; 7 043 filas y 21 columnas).
- **Descargado de:** <https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv> (repositorio `IBM/telco-customer-churn-on-icp4d`).
- **Copia equivalente:** Kaggle, `blastchar/telco-customer-churn` (archivo `WA_Fn-UseC_-Telco-Customer-Churn.csv`).
- **SHA-256 del archivo guardado:** `16320c9c1ec72448db59aa0a26a0b95401046bef5d02fd3aeb906448e3055e91`
- **Licencia:** Apache License 2.0 (según el repositorio de IBM, creado en 2019).

El archivo se guarda **sin modificar**. La traducción al español y la limpieza se hacen en memoria (`src/churn/datos.py` y la celda de carga de cada notebook).

## Cómo se obtienen

Los notebooks leen primero `data/Telco-Customer-Churn.csv`; si no existe (por ejemplo, en Google Colab) lo descargan de la URL anterior. Para descargarlo a mano:

```bash
curl -L -o data/Telco-Customer-Churn.csv https://raw.githubusercontent.com/IBM/telco-customer-churn-on-icp4d/master/data/Telco-Customer-Churn.csv
```

## Diccionario de variables (en español)

| Original | En español | Tipo | Valores / descripción |
|---|---|---|---|
| `customerID` | `id_cliente` | identificador | No se usa para modelar |
| `gender` | `genero` | categórica | Hombre / Mujer |
| `SeniorCitizen` | `adulto_mayor` | binaria | Sí / No (65 años o más) |
| `Partner` | `tiene_pareja` | binaria | Sí / No |
| `Dependents` | `tiene_dependientes` | binaria | Sí / No |
| `tenure` | `antiguedad_meses` | numérica | Meses como cliente (0–72) |
| `PhoneService` | `servicio_telefonico` | binaria | Sí / No |
| `MultipleLines` | `lineas_multiples` | categórica | Sí / No / Sin teléfono |
| `InternetService` | `servicio_internet` | categórica | DSL / Fibra óptica / Sin internet |
| `OnlineSecurity` | `seguridad_en_linea` | categórica | Sí / No / Sin internet |
| `OnlineBackup` | `respaldo_en_linea` | categórica | Sí / No / Sin internet |
| `DeviceProtection` | `proteccion_dispositivo` | categórica | Sí / No / Sin internet |
| `TechSupport` | `soporte_tecnico` | categórica | Sí / No / Sin internet |
| `StreamingTV` | `tv_streaming` | categórica | Sí / No / Sin internet |
| `StreamingMovies` | `peliculas_streaming` | categórica | Sí / No / Sin internet |
| `Contract` | `tipo_contrato` | categórica | Mensual / Un año / Dos años |
| `PaperlessBilling` | `factura_electronica` | binaria | Sí / No |
| `PaymentMethod` | `metodo_pago` | categórica | Cheque electrónico / Cheque por correo / Transferencia bancaria (automática) / Tarjeta de crédito (automática) |
| `MonthlyCharges` | `cargo_mensual` | numérica | USD por mes |
| `TotalCharges` | `cargo_total` | numérica | USD acumulados (viene como texto) |
| `Churn` | `abandona` | **objetivo** | 1 = abandonó, 0 = se quedó |

## Calidad de los datos (de `resultados/01_resumen_datos.json`)

- 11 valores en blanco en `TotalCharges`, todos de clientes con 0 meses de antigüedad. Se convierten a `NaN`; la imputación se hace dentro del Pipeline.
- Sin identificadores duplicados. 22 filas iguales si se ignora el identificador (clientes distintos con el mismo perfil; se conservan).
- Abandono: 1 869 de 7 043 clientes (26.5 %).

## Qué se excluyó del modelo y por qué

- **`id_cliente`:** es un identificador; no describe al cliente y solo permitiría memorizar.
- **`cargo_total`:** correlación de 0.9996 con `antiguedad_meses × cargo_mensual`. Es redundante y añadirla no mejora el modelo de forma consistente (`resultados/02_prueba_cargo_total.csv`). No es fuga de datos en sentido estricto, porque se conoce al momento de predecir.
