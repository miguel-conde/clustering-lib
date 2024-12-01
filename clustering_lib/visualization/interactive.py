# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# clustering_lib/visualization/interactive.py

import numpy as np
import matplotlib.pyplot as plt
from clustering_lib.algorithms.kmeans import KMeansClusterer
from clustering_lib.evaluation.metrics import silhouette_score
from clustering_lib.preprocessing.dimensionality_reduction import PCA
from ipywidgets import interact, IntSlider

def interactive_clustering(X, max_k=10):
    """
    Crea una visualización interactiva para explorar diferentes valores de k.

    Parameters:
        X (array-like): Datos de entrada.
        max_k (int): Número máximo de clusters.
    """
    # Reducir dimensionalidad para visualización
    pca = PCA(n_components=2)
    X_reduced = pca.fit_transform(X)

    def update(k):
        kmeans = KMeansClusterer(n_clusters=k)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)

        plt.figure(figsize=(8, 4))
        plt.scatter(X_reduced[:, 0], X_reduced[:, 1], c=labels, cmap='viridis', alpha=0.7)
        plt.title(f'Clustering con k={k}, Coeficiente de Silueta={score:.3f}')
        plt.xlabel('Componente Principal 1')
        plt.ylabel('Componente Principal 2')
        plt.show()

    interact(update, k=IntSlider(min=2, max=max_k, step=1, value=2))
