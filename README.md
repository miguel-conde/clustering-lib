# Clustering Lib

Una biblioteca de clustering en Python que implementa múltiples algoritmos y herramientas para el análisis y visualización de datos.

## Características

- **Algoritmos de Clustering**:
  - K-Means
  - Clustering Jerárquico (aglomerativo)
  - DBSCAN
  - Clustering Automatizado (AutoClustering)
  - (Agregar más algoritmos en el futuro)

- **Preprocesamiento**:
  - Escalado y normalización de datos
  - Reducción de dimensionalidad (PCA)

- **Evaluación**:
  - Métricas como coeficiente de silueta, índice de Calinski-Harabasz, etc.
  - Herramientas para determinar el número óptimo de clusters:
    - Método del Codo (Elbow Method)
    - Análisis del Coeficiente de Silueta
    - Estadística Gap (Gap Statistic)
    - Validación Cruzada Interna
    - Integración de múltiples métricas en un solo gráfico

- **Visualización**:
  - Gráficos 2D y 3D
  - Dendrogramas
  - Visualizaciones Interactivas para explorar diferentes valores de k

## Instalación

Puedes instalar la biblioteca directamente desde PyPI (una vez que la hayas publicado):

```bash
pip install clustering-lib
```

O puedes instalarla desde el repositorio:

```bash
git clone https://github.com/tu_usuario/clustering-lib.git
cd clustering-lib
pip install .
```

## Requisitos

+ Python 3.7 o superior
+ numpy
+ scipy
+ scikit-learn
+ matplotlib
+ ipywidgets (para visualizaciones interactivas)

## Uso Básico

Aquí hay un ejemplo de cómo utilizar la biblioteca para realizar clustering con K-Means en el conjunto de datos Iris:


```python
from clustering_lib.algorithms.auto_clustering import AutoClustering
from clustering_lib.preprocessing.scaling import StandardScaler
from clustering_lib.evaluation.selection import plot_combined_metrics
from clustering_lib.datasets.load_datasets import load_iris
from clustering_lib.visualization.plots import plot_clusters_2d
from clustering_lib.preprocessing.dimensionality_reduction import PCA

# Cargar datos
X, y = load_iris()

# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Análisis de métricas combinadas para determinar k óptimo
plot_combined_metrics(X_scaled, max_k=10)

# Clustering automatizado
auto_cluster = AutoClustering(method='kmeans', max_k=10, criterion='silhouette', random_state=42)
labels = auto_cluster.fit_predict(X_scaled)
print(f"Número óptimo de clusters: {auto_cluster.optimal_k}")

# Reducción de dimensionalidad para visualización
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Visualización de los clusters
plot_clusters_2d(X_reduced, labels)
```

También puedes utilizar las visualizaciones interactivas para explorar diferentes valores de $k$:
    
```python
from clustering_lib.visualization.interactive import interactive_clustering

# Ejecutar en una celda de Jupyter Notebook
interactive_clustering(X_scaled, max_k=10)
```

## Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar la biblioteca, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama para tu función o corrección de errores (git checkout -b feature/nueva-funcionalidad).
3. Realiza tus cambios y haz commits descriptivos.
4. Envía un pull request a la rama main del repositorio original.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo [LICENSE](LICENSE) para más detalles.

## Contacto

Para preguntas o soporte, puedes contactarme en miguelco2000@gmail.com.
