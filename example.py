# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
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
