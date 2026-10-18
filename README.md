# Predicción de rotación de productos: clustering y clasificación

Proyecto Final · Data Science II: Machine Learning para la Ciencia de Datos
**Autora:** Vicleniz Luque

## Pregunta de negocio
¿Qué hace que un producto rote mucho o poco? Con datos reales de ventas de una empresa (enero a septiembre de 2026) se calculó la rotación de 1.269 productos, se los segmentó con K-Means y se entrenó un modelo que predice si un producto será de **Alta** o **Baja** rotación.

## Enlaces
- Notebook en Colab: https://colab.research.google.com/drive/1AGm0OtSCx6kCU24tLNewUWIrPj7XWDxW
- Dataset final (anonimizado): https://docs.google.com/spreadsheets/d/1QbBgeodmfoMpiWJ1LUx0GgAqImKYIjRz

## Contenido del repositorio
- `proyecto_final.ipynb`: notebook con todo el pipeline ejecutado (EDA, limpieza, clustering, clasificación, interpretabilidad y conclusiones).
- `requirements.txt`: dependencias.
- `README.md`: este archivo.

## Datos y preparación
- Se cruzaron por SKU tres fuentes: reporte de ventas, maestro de productos y publicaciones de MercadoLibre.
- Solo se usaron ventas aprobadas. Se excluyeron los productos sin stock, porque sin stock no se puede calcular la rotación (de 2.079 productos con ventas quedaron 1.282).
- Varias marcas cargan el costo en dólares: se detectaron por la relación precio/costo (mayor a 20) y se convirtieron con el dólar oficial de $1.540.
- **Variable objetivo:** rotación mensual = unidades vendidas ÷ stock (~8,5 meses). Alta o Baja según la mediana, por lo que las clases quedan 50/50 por construcción.
- Las marcas están anonimizadas (Marca_001, Marca_002, etc.) por confidencialidad. El dataset no contiene datos personales de clientes.

## Metodología
1. **EDA y limpieza:** 1.282 filas, 1.269 tras eliminar 13 sin costo. Imputación de peso y comisión con la mediana. Los outliers se conservaron por ser productos reales.
2. **Clustering:** K-Means sobre precio, margen, stock y transacciones estandarizados. Se descartó K=2 (aislaba 27 productos atípicos) y se eligió K=3. Silhouette 0,29.
3. **Clasificación:** se excluyeron stock y transacciones para que el modelo no replique la fórmula de la rotación. División 80/20 estratificada, validación cruzada de 5 vueltas y ajuste de hiperparámetros con GridSearchCV.
4. **Interpretabilidad:** coeficientes de la regresión logística y valores SHAP.

## Resultados
Validación cruzada (5 vueltas):

| Modelo | Accuracy | F1 |
|---|---|---|
| Regresión Logística | 0,749 | 0,728 |
| Gradient Boosting | 0,731 | 0,707 |
| Random Forest | 0,726 | 0,716 |

Las diferencias son de unos ±2 puntos (empate técnico). Se eligió la **Regresión Logística** por ser la más simple e interpretable.

Modelo final en test (254 productos): accuracy 0,768 y ROC-AUC 0,808. Detecta el 84% de los productos de Baja rotación y el 69% de los de Alta; cuando predice Alta, acierta el 81%.

## Cómo reproducirlo
**En Colab (recomendado):** abrir el notebook y ejecutar *Entorno de ejecución → Ejecutar todas*. El dataset se descarga solo desde Google Sheets.

**En local:**
```bash
pip install -r requirements.txt
jupyter notebook proyecto_final.ipynb
```
Requiere Python 3.10 o superior y conexión a internet para descargar el dataset.

## Limitaciones
- Se usó el stock actual, no el histórico.
- La rotación es relativa a la mediana del catálogo, no un umbral de negocio.
- El test tiene 254 productos, por lo que la accuracy puede variar unos 5 puntos.
- Varias marcas tienen pocos productos, así que sus efectos son poco confiables.
- La conversión de costos en USD usa un dólar fijo.
