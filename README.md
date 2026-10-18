# Predicción de rotación de productos con clustering y clasificación

Proyecto final · Data Science II: Machine Learning para la Ciencia de Datos
**Autora:** Vicleniz Luque

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/viclenizdiaz-ctrl/Rotacion-productos/blob/main/notebook_rotacion_productos.ipynb)

## Problema

Con datos reales de ventas de una empresa (enero a septiembre de 2026) se busca responder: **¿qué hace que un producto rote mucho o poco?**

- **Tarea:** clasificación binaria (rotación **Alta** o **Baja**) + clustering exploratorio (K-Means).
- **Variable objetivo:** `rotacion_categoria`. Rotación = unidades vendidas ÷ stock; Alta/Baja según la mediana del catálogo (1.269 productos).
- **Métricas de éxito:** ROC-AUC (principal), recall de la clase Baja y F1. Accuracy como referencia.
- **Sin fuga de datos:** `stock_consolidado` y `transacciones` se usan solo para el clustering, no para clasificar.

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `notebook_rotacion_productos.ipynb` | Notebook completo: formulación, EDA, clustering, ingeniería de atributos, modelos, SHAP, conclusiones y presentación |
| `requirements.txt` | Dependencias |
| `README.md` | Este archivo |

## Datos

- **Dataset final:** [Google Sheets](https://docs.google.com/spreadsheets/d/1QbBgeodmfoMpiWJ1LUx0GgAqImKYIjRz/edit?gid=1209389706#gid=1209389706) (hoja `Dataset_Final`). El notebook lo descarga automáticamente.
- **Cómo se construyó:** se cruzaron por SKU tres fuentes de la empresa (ventas aprobadas, maestro de productos y publicaciones de MercadoLibre). Los costos en USD se convirtieron con un dólar fijo de $1.540. Las marcas están anonimizadas (`Marca_001`, `Marca_002`, ...) por confidencialidad.
- **Columnas principales:** `costo`, `precio_unitario_promedio`, `margen_pct`, `comision_pct`, `peso_g`, `categoria`, `marca_anon`, `tipo_publicacion`, `stock_consolidado`, `transacciones`, `unidades_vendidas`, `rotacion_mensual`, `rotacion_categoria`.

## Cómo reproducirlo

**Opción 1 (recomendada): Google Colab.** Click en el botón "Abrir en Colab" y luego *Entorno de ejecución → Ejecutar todas*. No hace falta subir archivos.

**Opción 2: local.**
```bash
git clone https://github.com/viclenizdiaz-ctrl/Rotacion-productos.git
cd Rotacion-productos
pip install -r requirements.txt
jupyter notebook notebook_rotacion_productos.ipynb
```
Semilla fija (`RANDOM_SEED = 42`) en todo el notebook.

## Resumen del método

1. **EDA:** nulos, outliers (regla IQR), distribuciones, correlaciones.
2. **Clustering:** K-Means (K=3) sobre precio, margen, stock y transacciones estandarizados; PCA solo para visualizar.
3. **Ingeniería de atributos:** logaritmos de precio, costo y peso; tamaño de marca; indicador de "sin categoría"; agrupación de marcas con menos de 10 productos; one-hot encoding.
4. **Modelos:** Regresión Logística, Random Forest y XGBoost, con GridSearchCV / RandomizedSearchCV (5 vueltas, métrica ROC-AUC).
5. **Selección:** por AUC de validación cruzada, interpretabilidad y costo de los errores.
6. **Explicabilidad:** SHAP.

## Principales resultados

Los valores finales están en el notebook (tabla comparativa de la sección "Ingeniería de atributos, optimización de hiperparámetros y comparación final").

- Los productos de Alta rotación son más caros y con menor margen; los de Baja, más baratos y con mayor margen.
- El margen es la variable numérica más influyente: a mayor margen, menor probabilidad de Alta rotación.
- Limitaciones: stock actual (no histórico), rotación relativa a la mediana, test de solo 254 productos, marcas con pocos productos.
