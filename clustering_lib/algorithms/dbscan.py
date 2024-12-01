# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# clustering_lib/algorithms/dbscan.py

import numpy as np
from clustering_lib.base import ClustererBase
from clustering_lib.utils.helpers import validate_input
from sklearn.neighbors import NearestNeighbors


class DBSCANClusterer(ClustererBase):
    """
    Implementación del algoritmo DBSCAN para clustering basado en densidad.
    """

    def __init__(self, eps=0.5, min_samples=5, metric='euclidean'):
        """
        Inicializa el algoritmo DBSCANClusterer.

        Parameters:
            eps (float): Distancia máxima entre dos muestras para considerarlas como vecinas.
            min_samples (int): Número mínimo de muestras para formar un cluster denso.
            metric (str): Métrica de distancia a utilizar.
        """
        self.eps = eps
        self.min_samples = min_samples
        self.metric = metric
        self.labels_ = None
        self.core_sample_indices_ = None

    def fit(self, X):
        """
        Ajusta el modelo DBSCAN a los datos X.

        Parameters:
            X (array-like): Datos de entrada de forma (n_samples, n_features).
        """
        X = validate_input(X)
        n_samples = X.shape[0]
        labels = -np.ones(n_samples, dtype=int)
        visited = np.zeros(n_samples, dtype=bool)
        cluster_id = 0

        nbrs = NearestNeighbors(radius=self.eps, metric=self.metric)
        nbrs.fit(X)
        neighbors = nbrs.radius_neighbors(X, return_distance=False)

        for i in range(n_samples):
            if not visited[i]:
                visited[i] = True
                neighbors_i = neighbors[i]
                if len(neighbors_i) >= self.min_samples:
                    labels = self._expand_cluster(labels, i, neighbors, cluster_id, visited)
                    cluster_id += 1

        self.labels_ = labels
        self.core_sample_indices_ = np.where(labels != -1)[0]

    def _expand_cluster(self, labels, i, neighbors, cluster_id, visited):
        """
        Expande el cluster a partir del punto dado.

        Parameters:
            labels (ndarray): Etiquetas de cluster.
            i (int): Índice del punto inicial.
            neighbors (list of arrays): Vecinos de cada punto.
            cluster_id (int): ID del cluster actual.
            visited (ndarray): Array booleano de puntos visitados.

        Returns:
            labels (ndarray): Etiquetas de cluster actualizadas.
        """
        labels[i] = cluster_id
        seeds = list(neighbors[i])
        while seeds:
            current_point = seeds.pop()
            if not visited[current_point]:
                visited[current_point] = True
                neighbors_current = neighbors[current_point]
                if len(neighbors_current) >= self.min_samples:
                    seeds.extend([n for n in neighbors_current if not visited[n]])
            if labels[current_point] == -1:
                labels[current_point] = cluster_id
        return labels

    def predict(self, X):
        """
        Como DBSCAN es un algoritmo no predictivo, este método lanza una excepción.

        Raises:
            NotImplementedError: Siempre, ya que el modelo no puede predecir nuevos datos.
        """
        raise NotImplementedError("DBSCAN no soporta predicciones en nuevos datos.")

    def fit_predict(self, X):
        """
        Ajusta el modelo y devuelve las etiquetas de cluster.

        Parameters:
            X (array-like): Datos de entrada.

        Returns:
            labels (ndarray): Etiquetas de cluster asignadas a cada muestra.
        """
        self.fit(X)
        return self.labels_
