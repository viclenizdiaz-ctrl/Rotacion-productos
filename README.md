# Predicción de rotación de productos con clustering y clasificación

Proyecto final · Data Science II: Machine Learning para la Ciencia de Datos
**Autora:** Vicleniz Luque

[![Abrir en Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/viclenizdiaz-ctrl/Rotacion-productos/blob/main/notebook_rotacion_productos_liviano.ipynb)

> **Cómo leer este proyecto.** Este README contiene el informe completo en texto, con los resultados. Archivos del repositorio:
> - `notebook_rotacion_productos_liviano.ipynb`: notebook ejecutado de principio a fin, con todas las tablas y salidas de texto (sin imágenes, para que cargue rápido).
> - `predicción_de_rotación_de_productos_con_clustering_y_clasificación__DATA_II (2).ipynb`: el mismo notebook ejecutado, con todos los gráficos.
> - `requirements.txt`: dependencias.
> - > - `informe_rotacion_productos.pdf`: copia completa del notebook en PDF, con código, tablas y gráficos.

## 1. Formulación del problema

- **Pregunta:** ¿qué hace que un producto rote mucho o poco?
- **Tarea:** clasificación binaria supervisada (rotación **Alta** o **Baja**), más un clustering exploratorio (K-Means).
- **Variable objetivo:** `rotacion_categoria`. Rotación = unidades vendidas ÷ stock (enero a septiembre de 2026). Es Alta o Baja según la mediana del catálogo, por eso las clases quedan balanceadas (50,3% Baja / 49,7% Alta).
- **Predictores:** solo características conocidas antes de comprar: precio, costo, margen, comisión, peso, marca, categoría y tipo de publicación. **Se excluyen `stock_consolidado` y `transacciones`** porque la rotación se calcula con ellas (evita fuga de datos).
- **Métricas de éxito:** ROC-AUC (principal, mide cómo ordena los productos sin depender de un umbral), recall de la clase Baja y F1. La accuracy es referencia (el azar da 50%).
- **Relevancia:** comprar de más un producto de baja rotación inmoviliza capital y genera sobrestock.

## 2. Datos

- **Dataset final:** [Google Sheets](https://docs.google.com/spreadsheets/d/1QbBgeodmfoMpiWJ1LUx0GgAqImKYIjRz/edit?gid=1209389706#gid=1209389706) (hoja `Dataset_Final`). El notebook lo descarga solo.
- **Construcción:** se cruzaron por SKU tres fuentes de la empresa (ventas aprobadas, maestro de productos y publicaciones de MercadoLibre). Los costos en USD se convirtieron con un dólar fijo de $1.540. Las marcas están anonimizadas.
- **Limpieza:** 1.282 filas originales, 1.269 después de eliminar las que no tenían costo ni margen. Peso y comisión se imputaron con la mediana; el tipo de publicación faltante se marcó como "Sin dato".
- El 6,3% de los productos no tiene categoría porque ese dato solo existe para los publicados en MercadoLibre.

## 3. Hallazgos del EDA

- **Margen vs. rotación:** los productos de Baja rotación tienen más margen (mediana ~55-60%) que los de Alta (~45%).
- **Precio vs. rotación:** los de Alta rotación son más caros (mediana ~$100.000) que los de Baja (~$60.000-70.000).
- **Patrón:** los de Alta rotación compiten por volumen (más caros, menor margen unitario); los de Baja son de nicho (más baratos, mayor margen).
- **Categorías:** Hogar/Muebles y Griferías de Baño concentran la mayor proporción de productos de Alta rotación.
- Ninguna variable por sí sola separa bien las clases, por lo que se necesita un modelo que las combine.

## 4. Clustering (K-Means)

K-Means sobre precio, margen, stock y transacciones estandarizados. Se descartó K=2 porque aislaba solo 27 productos atípicos (silhouette inflado de 0,75). Con K=3: silhouette 0,293, Davies-Bouldin 1,166, Calinski-Harabasz 312,9 (separación moderada). El PCA se usó solo para graficar (58,1% de la varianza).

| Cluster | Productos | Precio medio | Margen | Stock medio | Transacciones | Perfil |
|---|---|---|---|---|---|---|
| 0 | 624 | $81.848 | 61% | 677 | 13,5 | Precio medio, margen alto. 33% Alta rotación |
| 1 | 27 | $36.756 | 61% | 8.341 | 136,4 | Baratos, alto volumen y mucho stock. 85% Baja rotación (posible sobrestock) |
| 2 | 618 | $155.057 | 39% | 206 | 13,1 | Precio alto, margen bajo. 68% Alta rotación |

## 5. Clasificación

### Ingeniería de atributos
- `log_precio`, `log_costo`, `log_peso`: precio, costo y peso tienen distribuciones muy asimétricas con outliers.
- `marca_n_productos`: tamaño de la marca en el catálogo, en una sola columna en lugar de decenas de dummies.
- `sin_categoria`: marca los productos no publicados en MercadoLibre (dato faltante estructural).
- Además: marcas con menos de 10 productos agrupadas en "Otras", one-hot encoding y estandarización dentro de un pipeline (el escalador aprende solo del train en cada vuelta).

### Comparación inicial (validación cruzada de 5 vueltas, atributos originales)

| Modelo | Accuracy | F1 |
|---|---|---|
| Regresión Logística | 0,749 | 0,728 |
| Gradient Boosting | 0,731 | 0,707 |
| Random Forest | 0,726 | 0,716 |

### Optimización de hiperparámetros y comparación final
GridSearchCV (Regresión Logística) y RandomizedSearchCV (Random Forest, 15 combinaciones; XGBoost, 25 combinaciones), 5 vueltas, métrica ROC-AUC. Partición 80/20 estratificada (254 productos de test), semilla 42.

| Modelo | AUC (CV) | Accuracy | Precision (Alta) | Recall (Alta) | Recall (Baja) | F1 (Alta) | ROC-AUC test |
|---|---|---|---|---|---|---|---|
| Random Forest v3 | 0,797 | 0,772 | 0,833 | 0,675 | 0,867 | 0,746 | 0,824 |
| **Regresión Logística v2 (elegido)** | 0,796 | 0,768 | 0,813 | 0,690 | 0,844 | 0,747 | 0,808 |
| Regresión Logística v3 | 0,794 | 0,768 | 0,825 | 0,675 | 0,859 | 0,742 | 0,810 |
| XGBoost v3 | 0,794 | 0,756 | 0,802 | 0,675 | 0,836 | 0,733 | 0,808 |

Mejores hiperparámetros: Regresión Logística v3 `C=1`; Random Forest `n_estimators=600, max_depth=12, min_samples_leaf=3, max_features='sqrt'`; XGBoost `n_estimators=300, max_depth=5, learning_rate=0.03, subsample=0.85, colsample_bytree=0.8, min_child_weight=5, reg_lambda=10`.

### Selección del modelo
Criterios, en orden: (1) ROC-AUC en validación cruzada (no en test, para no elegir mirando el examen), considerando empate una mejora menor a 0,02; (2) interpretabilidad para el equipo de compras; (3) costo de los errores (detectar los productos de Baja rotación); (4) simplicidad y estabilidad.

Los cuatro modelos quedaron empatados (AUC de CV entre 0,794 y 0,797). Los atributos nuevos no mejoraron al modelo lineal y los modelos de árboles no superan el umbral de 0,02. El Random Forest tiene el mayor AUC en test (0,824), pero su ventaja en validación cruzada es de 0,001 y con 254 productos de test esa diferencia es ruido. **Se elige la Regresión Logística v2** por ser la más simple e interpretable.

### Resultado final (Regresión Logística v2, test de 254 productos)
Accuracy 0,768 · ROC-AUC 0,808 (0,752 en validación cruzada).

| | Predijo Baja | Predijo Alta |
|---|---|---|
| **Real Baja** | 108 | 20 |
| **Real Alta** | 39 | 87 |

Detecta el 84% de los productos de Baja rotación y el 69% de los de Alta; cuando predice Alta acierta el 81%.

### Explicabilidad (SHAP y coeficientes)
- **Margen:** es la variable numérica más influyente (coeficiente estandarizado -0,50): a mayor margen, menor probabilidad de Alta rotación.
- **Marca:** Marca_025 (194 productos, 21% Alta) y Marca_068 (43 productos, 23%) tienden a Baja rotación. Esto puede reflejar la política de compra (sobrestock) más que la demanda.
- **Publicación Premium:** 45% de Alta rotación contra 53% en las Clásicas; no mejora la rotación.
- Se calculó SHAP para el modelo lineal elegido y también para XGBoost, para contrastar ambas explicaciones.

## 6. Conclusiones y recomendaciones

1. Revisar el stock de los 27 productos de alto volumen: venden mucho pero el 85% figura como Baja rotación por sobrestock.
2. Usar el modelo como filtro antes de comprar: marca los productos con probable baja rotación, sobre todo los de margen alto o de marcas con tendencia a Baja.
3. Evaluar si la publicación Premium se justifica, ya que no mejora la rotación.
4. Estandarizar la categorización de productos en todos los canales de venta.

**Limitaciones:** se usó el stock actual y no el histórico; la rotación es relativa a la mediana del catálogo, no un umbral de negocio; el test tiene 254 productos (la accuracy puede variar unos 5 puntos); varias marcas tienen pocos productos; la conversión de costos en USD usa una regla (precio/costo > 20) y un dólar fijo; el período es de 8,5 meses. Los resultados son asociaciones, no relaciones causales.

## 7. Cómo reproducirlo

**Colab (recomendado):** botón "Abrir en Colab" y luego *Entorno de ejecución → Ejecutar todas*. No hace falta subir archivos.

**Local:**
```bash
git clone https://github.com/viclenizdiaz-ctrl/Rotacion-productos.git
cd Rotacion-productos
pip install -r requirements.txt
jupyter notebook notebook_rotacion_productos_liviano.ipynb
```

| Archivo | Descripción |
|---|---|
| `notebook_rotacion_productos_liviano.ipynb` | Notebook ejecutado, sin imágenes (es el que abre el botón de Colab) |
| `predicción_de_rotación_de_productos_con_clustering_y_clasificación__DATA_II (2).ipynb` | Notebook ejecutado, con todos los gráficos |
| `requirements.txt` | Dependencias |
| `README.md` | Este archivo |
| `informe_rotacion_productos.pdf` | Informe completo en PDF, con código, tablas y gráficos |
