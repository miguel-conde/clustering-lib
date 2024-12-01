from clustering_lib.algorithms.auto_clustering import AutoClustering
from clustering_lib.datasets.load_datasets import load_iris
from clustering_lib.preprocessing.scaling import StandardScaler
from clustering_lib.evaluation.selection import plot_combined_metrics
from clustering_lib.visualization.plots import plot_clusters_2d
from clustering_lib.preprocessing.dimensionality_reduction import PCA

# Cargar datos
X, y = load_iris()

# Preprocesamiento
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Análisis de métricas combinadas
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
