# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# clustering_lib/algorithms/kmeans.py

import numpy as np
from sklearn.base import BaseEstimator
from clustering_lib.base import ClustererBase


class KMeansClusterer(ClustererBase, BaseEstimator):
    """
    Implementación del algoritmo K-Means clustering.
    """

    def __init__(self, n_clusters=8, max_iter=300, tol=1e-4, random_state=None):
        """
        Inicializa el algoritmo KMeansClusterer.

        Parameters:
            n_clusters (int): Número de clusters a formar.
            max_iter (int): Número máximo de iteraciones.
            tol (float): Tolerancia para declarar convergencia.
            random_state (int): Semilla para el generador de números aleatorios.
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.tol = tol
        self.random_state = random_state
        self.cluster_centers_ = None
        self.labels_ = None
        self.inertia_ = None

    def fit(self, X):
        """
        Ajusta el modelo K-Means a los datos X.

        Parameters:
            X (array-like): Datos de entrada de forma (n_samples, n_features).
        """
        X = np.asarray(X)
        n_samples, n_features = X.shape

        rng = np.random.default_rng(self.random_state)
        self.cluster_centers_ = X[rng.choice(n_samples, self.n_clusters, replace=False)]

        for i in range(self.max_iter):
            # Asignar muestras al centroide más cercano
            distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
            labels = np.argmin(distances, axis=1)

            # Recalcular centroides
            new_centers = np.array([
                X[labels == j].mean(axis=0) if np.any(labels == j) else self.cluster_centers_[j]
                for j in range(self.n_clusters)
            ])

            # Verificar convergencia
            center_shift = np.linalg.norm(self.cluster_centers_ - new_centers, axis=1).sum()
            self.cluster_centers_ = new_centers

            if center_shift <= self.tol:
                break

        self.labels_ = labels
        self.inertia_ = np.sum((X - self.cluster_centers_[labels]) ** 2)

    def predict(self, X):
        """
        Asigna etiquetas a las muestras en X.

        Parameters:
            X (array-like): Datos de entrada de forma (n_samples, n_features).

        Returns:
            labels (array): Etiquetas de cluster asignadas a cada muestra.
        """
        X = np.asarray(X)
        distances = np.linalg.norm(X[:, np.newaxis] - self.cluster_centers_, axis=2)
        return np.argmin(distances, axis=1)
