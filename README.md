# Clustering Lib

Una biblioteca de clustering en Python que implementa múltiples algoritmos y herramientas para el análisis y visualización de datos.

## Características

- **Algoritmos de Clustering**:
  - K-Means
  - Clustering Jerárquico (aglomerativo)
  - DBSCAN
  - (Agregar más algoritmos en el futuro)

- **Preprocesamiento**:
  - Escalado y normalización de datos
  - Reducción de dimensionalidad (PCA)

- **Evaluación**:
  - Métricas como coeficiente de silueta, índice de Calinski-Harabasz, etc.

- **Visualización**:
  - Gráficos 2D y 3D
  - Dendrogramas

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

## Uso Básico

Aquí hay un ejemplo de cómo utilizar la biblioteca para realizar clustering con K-Means en el conjunto de datos Iris:


```python
from clustering_lib.algorithms.kmeans import KMeansClusterer
from clustering_lib.preprocessing.scaling import StandardScaler
from clustering_lib.evaluation.metrics import silhouette_score
from clustering_lib.datasets.load_datasets import load_iris
from clustering_lib.visualization.plots import plot_clusters_2d
from clustering_lib.preprocessing.dimensionality_reduction import PCA

# Cargar datos
X, y = load_iris()

# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Reducción de dimensionalidad
pca = PCA(n_components=2)
X_reduced = pca.fit_transform(X_scaled)

# Clustering
kmeans = KMeansClusterer(n_clusters=3, random_state=42)
labels = kmeans.fit_predict(X_scaled)

# Evaluación
score = silhouette_score(X_scaled, labels)
print(f"Coeficiente de silueta: {score:.3f}")

# Visualización
plot_clusters_2d(X_reduced, labels)
```

## Contribuciones

¡Las contribuciones son bienvenidas! Si deseas mejorar la biblioteca, por favor sigue estos pasos:

1. Haz un fork del repositorio.
2. Crea una nueva rama para tu función o corrección de errores (git checkout -b feature/nueva-funcionalidad).
3. Realiza tus cambios y haz commits descriptivos.
4. Envía un pull request a la rama main del repositorio original.

## Licencia

Este proyecto está bajo la Licencia MIT. Consulta el archivo LICENSE para más detalles.

## Contacto

Para preguntas o soporte, puedes contactarme en tu.email@example.com.
