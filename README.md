# 💶 Predicción de Tesorería Corporativa con IA (Cash Flow Forecasting)

## 📌 Descripción del Proyecto
La gestión de la liquidez es el pilar fundamental de cualquier departamento financiero. Este proyecto desarrolla un **Gemelo Digital** de la tesorería de una empresa *asset-heavy* (Real Estate / Sector Educativo) y utiliza algoritmos de Inteligencia Artificial (**Facebook Prophet**) para predecir el flujo de caja a 60 días, superando las limitaciones de los modelos estáticos en Excel.

## 🏢 El Caso de Negocio (Business Case)
Se ha modelado una empresa que gestiona 10 residencias de estudiantes (3.000 camas) en España, sujeta a una fuerte estacionalidad y a una estructura de costes rígida.

### Reglas del Modelo Financiero Inyectadas:
* **Ingresos Híbridos:** * *Septiembre - Junio:* Cobro masivo de mensualidades (días 1 al 5).
  * *Julio - Agosto:* Transición a modelo "albergue" con ingresos diarios y picos en fines de semana.
* **Estructura de Costes (OpEx & Deuda):**
  * Pago de nóminas (día 28).
  * Fuerte carga de deuda inmobiliaria / Leasing (día 5).
  * Liquidación trimestral de IVA (día 20 de los meses de cierre).
* **Retribución al Accionista:** Salida masiva de caja por pago de dividendos (30 de junio).

## 📊 Conclusiones y Resultados del Modelo
El algoritmo de predicción de series temporales ha sido capaz de interiorizar las reglas de negocio, arrojando los siguientes *insights*:

1. **Predicción a 60 días (Liquidez Asegurada):** El modelo proyecta la liquidez para los próximos dos meses manteniendo el saldo sólidamente por encima de los 4,5M€, confirmando una posición de caja segura para afrontar obligaciones a corto plazo.
2. **Crecimiento Estructural (Trend):** Se observa una tendencia incremental sostenida (de 3,8M€ a 4,8M€ en dos años), indicando que el modelo de negocio es rentable y genera *Free Cash Flow* de forma constante.
3. **El Valle Estival (Yearly Seasonality):** El algoritmo detecta de forma autónoma una contracción drástica de la liquidez a finales de junio. Esto responde matemáticamente a la salida de 1M€ en dividendos combinada con la transición al modelo estival (cese de cobro de rentas fijas).

## 🛠️ Tecnologías Utilizadas
* **Python 3.12**
* **Pandas & NumPy:** Ingeniería de datos y creación del dataset sintético financiero.
* **Prophet (Meta):** Modelado predictivo de series temporales.
* **Matplotlib:** Visualización de datos e intervalos de confianza.

## 🚀 Cómo ejecutar este proyecto
1. Clona este repositorio en tu máquina local.
2. Instala las dependencias necesarias: `pip install pandas numpy prophet matplotlib`
3. Ejecuta el script `generador_datos.py` para crear el dataset sintético (`tesoreria_residencias.csv`).
4. Abre y ejecuta el notebook `prediccion_tesoreria.ipynb` para visualizar el entrenamiento y las predicciones del modelo.