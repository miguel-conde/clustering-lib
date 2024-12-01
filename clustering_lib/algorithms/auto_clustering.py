# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# clustering_lib/algorithms/auto_clustering.py

import numpy as np
from clustering_lib.algorithms.kmeans import KMeansClusterer
from clustering_lib.evaluation.metrics import silhouette_score

class AutoClustering:
    """
    Automatiza la selección del número óptimo de clusters utilizando el criterio especificado.
    """

    def __init__(self, method='kmeans', max_k=10, criterion='silhouette', random_state=None):
        """
        Inicializa el AutoClustering.

        Parameters:
            method (str): Método de clustering ('kmeans' por defecto).
            max_k (int): Número máximo de clusters a probar.
            criterion (str): Criterio para seleccionar el k óptimo ('silhouette' o 'gap').
            random_state (int): Semilla para el generador de números aleatorios.
        """
        self.method = method
        self.max_k = max_k
        self.criterion = criterion
        self.random_state = random_state
        self.optimal_k = None
        self.labels_ = None

    def fit_predict(self, X):
        """
        Ajusta el modelo y devuelve las etiquetas de cluster.

        Parameters:
            X (array-like): Datos de entrada.

        Returns:
            labels (ndarray): Etiquetas de cluster asignadas a cada muestra.
        """
        scores = []
        Ks = range(2, self.max_k + 1)

        for k in Ks:
            if self.method == 'kmeans':
                clusterer = KMeansClusterer(n_clusters=k, random_state=self.random_state)
            else:
                raise ValueError("Método no soportado.")

            labels = clusterer.fit_predict(X)

            if self.criterion == 'silhouette':
                score = silhouette_score(X, labels)
            else:
                raise ValueError("Criterio no soportado.")

            scores.append(score)

        # Seleccionar k óptimo
        self.optimal_k = Ks[np.argmax(scores)]

        # Ajustar el modelo con k óptimo
        self.clusterer = KMeansClusterer(n_clusters=self.optimal_k, random_state=self.random_state)
        self.labels_ = self.clusterer.fit_predict(X)

        return self.labels_
