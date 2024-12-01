# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# clustering_lib/algorithms/hierarchical.py

import numpy as np
from clustering_lib.base import ClustererBase
from clustering_lib.utils.helpers import validate_input
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


class HierarchicalClusterer(ClustererBase):
    """
    Implementación del algoritmo de clustering jerárquico aglomerativo.
    """

    def __init__(self, n_clusters=2, linkage_method='ward', affinity='euclidean'):
        """
        Inicializa el algoritmo HierarchicalClusterer.

        Parameters:
            n_clusters (int): Número de clusters deseados.
            linkage_method (str): Método de enlace ('single', 'complete', 'average', 'ward').
            affinity (str): Métrica de distancia ('euclidean', 'cityblock', etc.).
        """
        self.n_clusters = n_clusters
        self.linkage_method = linkage_method
        self.affinity = affinity
        self.labels_ = None
        self.linkage_matrix_ = None

    def fit(self, X):
        """
        Ajusta el modelo a los datos X.

        Parameters:
            X (array-like): Datos de entrada de forma (n_samples, n_features).
        """
        X = validate_input(X)
        distance_matrix = pdist(X, metric=self.affinity)
        self.linkage_matrix_ = linkage(distance_matrix, method=self.linkage_method)
        self.labels_ = fcluster(self.linkage_matrix_, t=self.n_clusters, criterion='maxclust')

    def predict(self, X):
        """
        Como el clustering jerárquico no es predictivo, este método lanza una excepción.

        Raises:
            NotImplementedError: Siempre, ya que el modelo no puede predecir nuevos datos.
        """
        raise NotImplementedError("El clustering jerárquico no soporta predicciones en nuevos datos.")

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
