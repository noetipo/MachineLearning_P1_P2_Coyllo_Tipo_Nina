# Manual de la aplicación de escritorio

Alerta temprana de abandono de clientes (`app/app_escritorio.py`)

La aplicación estima la probabilidad de abandono de un cliente individual o de un archivo de clientes mediante el modelo final del proyecto (LightGBM optimizado, `app/modelo.joblib`).

## 1. Instalación paso a paso

Requisitos: Python 3.11 o superior con tkinter. El proyecto se ejecutó con Python 3.13.5.

1. Abra una terminal en la carpeta del proyecto.
2. Compruebe que su Python tiene tkinter:

   ```bash
   python3 -m tkinter        # debe abrir una ventana de prueba
   ```

   Si aparece `No module named '_tkinter'`, instálelo según su sistema:

   | Sistema | Cómo obtener tkinter |
   |---|---|
   | **Windows** | El instalador oficial de python.org lo incluye: marque *tcl/tk and IDLE* al instalar (o use *Modify* en el instalador). |
   | **macOS (Homebrew)** | `brew install python-tk@3.13` (ajuste la versión a la de su Python). Con pyenv, reinstale Python después de `brew install tcl-tk`. |
   | **Ubuntu / Debian** | `sudo apt install python3-tk` |
   | **Fedora** | `sudo dnf install python3-tkinter` |

3. Cree el entorno virtual e instale las dependencias:

   ```bash
   # macOS / Linux
   python3 -m venv .venv
   .venv/bin/pip install -r requirements.txt        # o solo app/requirements.txt para usar la app

   # Windows (PowerShell)
   py -3.13 -m venv .venv
   .venv\Scripts\pip install -r requirements.txt
   ```

4. Verifique que existe `app/modelo.joblib`. Si no existe, o si al abrir la app aparece un error al cargar el modelo (por versiones distintas de scikit-learn o lightgbm), regénérelo:

   ```bash
   .venv/bin/python scripts/entrenar.py
   ```

## 2. Cómo ejecutarla

Desde la raíz del proyecto:

```bash
.venv/bin/python app/app_escritorio.py            # macOS / Linux
.venv\Scripts\python app\app_escritorio.py        # Windows
```

La ventana es redimensionable y consta de tres pestañas.

## 3. Pestañas

### 3.1 Predicción individual

- En la parte superior, el selector **Caso de prueba** permite cargar uno de los ocho clientes reales de `data/prueba/casos_ejemplo.csv`. Al pulsar **Cargar caso**, el formulario se rellena, se estima el riesgo y se indica el valor real del cliente y si la predicción fue correcta.
- **Formulario** en tres recuadros, agrupados por bloque temático:
  - *Datos del cliente:* género (lista), adulto mayor, tiene pareja, tiene dependientes (casillas) y antigüedad en meses (selector numérico).
  - *Servicios contratados:* servicio telefónico (casilla), líneas múltiples, servicio de internet, seguridad en línea, respaldo en línea, protección de dispositivo, soporte técnico, TV y películas por streaming (listas).
  - *Contrato y pagos:* tipo de contrato, método de pago (listas), factura electrónica (casilla) y cargo mensual en USD (selector numérico).
- Las opciones de cada lista se obtienen del modelo guardado (categorías observadas durante el entrenamiento).
- Si elige *Servicio de internet = Sin internet*, los seis servicios que dependen de internet pasan a "Sin internet" y se bloquean; lo mismo con *Líneas múltiples* si desmarca el servicio telefónico. Así se evitan combinaciones imposibles.
- **Estimar riesgo** muestra:
  - la probabilidad de abandono (por ejemplo, `58.9%`);
  - el nivel de riesgo con color: **bajo** (verde, < 30 %), **medio** (ámbar, 30–50 %) o **alto** (rojo, ≥ 50 %; el 50 % es el umbral de decisión del modelo);
  - una recomendación breve (sin acción, seguimiento con beneficios de fidelización, o contacto pronto con una oferta de retención).
- **Limpiar** vuelve a los valores iniciales.
- Si un número no es válido (por ejemplo, letras en la antigüedad, o un cargo fuera de 0–500 USD), aparece un mensaje de error y la app sigue abierta.

### 3.2 Predicción por lote

1. Pulse **Cargar CSV de clientes…** y elija un archivo. Se aceptan dos formatos:
   - el formato original de IBM, con columnas en inglés (`customerID`, `gender`, `tenure`, …), por ejemplo el propio `data/Telco-Customer-Churn.csv`;
   - columnas ya en español (`id_cliente`, `genero`, `antiguedad_meses`, …).
   La columna `Churn`/`abandona` y `TotalCharges`/`cargo_total` pueden estar o no; se ignoran.
2. El diálogo de selección se abre en `data/prueba/`. La tabla muestra, por cliente: identificador, probabilidad, nivel (con color de fondo), contrato, antigüedad, cargo mensual, internet método de pago y, si el archivo la incluye, la columna *Real* con el resultado observado. Se presenta ordenada de mayor a menor riesgo; haga clic en cualquier encabezado para ordenar por esa columna (otro clic invierte el orden).
3. Debajo se resume cuántos clientes hay en cada nivel. Si el archivo incluye `abandona` o `Churn`, se muestran además la matriz de confusión, el recall, la precisión y la exactitud del lote. Con `data/prueba/clientes_test.csv` se obtiene VP = 301, FN = 73, FP = 292 y VN = 743 (recall 0.805), idéntico a la evaluación oficial del modelo.
4. **Exportar resultados…** guarda un CSV (UTF-8, se abre bien en Excel) con todas las variables, `prob_abandono` y `nivel_riesgo`.

Si al archivo le faltan columnas, la app indica cuáles (por ejemplo, *Faltan columnas obligatorias: tipo_contrato, metodo_pago*) y no se cierra. Lo mismo si hay una categoría desconocida (indica la columna, el valor y las opciones válidas).

### 3.3 Acerca de

Muestra el proyecto, el curso, el docente, los integrantes, el modelo usado con su fecha de entrenamiento y sus **métricas en el test**, leídas de `resultados/modelo_final.json` (recall 0.805, precisión 0.508, ROC-AUC 0.848, PR-AUC 0.667 y la matriz de confusión), y la advertencia de uso responsable.

> **Uso responsable.** El modelo está destinado a ofrecer acciones de retención. No debe usarse para negar servicios, subir precios ni tratar peor a ningún cliente. Se entrenó con un dataset de muestra de IBM sobre una empresa ficticia.

## 4. Empaquetar como ejecutable (opcional)

PyInstaller permite generar un ejecutable que no requiere instalar Python. No se generó en este proyecto; los comandos son:

```bash
.venv/bin/pip install pyinstaller

# macOS / Linux (separador ":" en --add-data)
.venv/bin/pyinstaller --name AlertaAbandono --windowed --noconfirm \
  --paths src \
  --add-data "app/modelo.joblib:app" \
  --add-data "resultados/modelo_final.json:resultados" \
  --collect-all lightgbm --collect-all sklearn \
  app/app_escritorio.py

# Windows (separador ";" en --add-data)
.venv\Scripts\pyinstaller --name AlertaAbandono --windowed --noconfirm --paths src ^
  --add-data "app\modelo.joblib;app" --add-data "resultados\modelo_final.json;resultados" ^
  --collect-all lightgbm --collect-all sklearn app\app_escritorio.py
```

El resultado queda en `dist/AlertaAbandono/`. Nota: la app busca el modelo con rutas relativas a la carpeta del proyecto (`src/churn/prediccion.py`, constantes `RUTA_MODELO` y `RUTA_INFO`); dentro de un ejecutable de PyInstaller esas rutas cambian (`sys._MEIPASS`), así que antes de distribuirlo hay que adaptar esas dos constantes y probar el ejecutable.

## 5. Solución de problemas

| Problema | Causa probable | Solución |
|---|---|---|
| `No module named '_tkinter'` | Python sin tkinter | Ver la tabla de la sección 1 |
| "Modelo no encontrado" | Falta `app/modelo.joblib` | `python scripts/entrenar.py` |
| Error al cargar el modelo / advertencias de versión | Versiones distintas de scikit-learn o lightgbm | `pip install -r requirements.txt` o reentrenar con `scripts/entrenar.py` |
| La ventana no aparece (macOS) | Sesión gráfica bloqueada o sin permisos | Desbloquear la pantalla y ejecutar desde una terminal de la sesión gráfica |

## 6. Pruebas

- **Automatizadas:** la lógica de predicción se prueba sin interfaz (`tests/test_prediccion.py`, `tests/test_datos_prueba.py`). La prueba de humo de la ventana (`tests/test_apps.py`, que ejecuta `tests/humo_app_escritorio.py`) se omite si no hay entorno gráfico.
- **Funcional:** `python scripts/capturas_app.py` abre la aplicación, ejecuta los escenarios con los archivos de `data/prueba/` y registra los resultados en `docs/evidencias/prueba_app_escritorio.md`. En macOS, si la terminal tiene permiso de *Grabación de pantalla* (Configuración del Sistema → Privacidad y seguridad), guarda también capturas en `docs/capturas/`.
