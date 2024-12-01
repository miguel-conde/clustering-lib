# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# tests/test_visualization.py

import unittest
import numpy as np
from clustering_lib.visualization.plots import plot_clusters_2d
from clustering_lib.datasets.load_datasets import load_iris
from clustering_lib.preprocessing.dimensionality_reduction import PCA
import matplotlib

# Usar backend 'Agg' para evitar que las pruebas intenten mostrar gráficos
matplotlib.use('Agg')


class TestVisualization(unittest.TestCase):
    def test_plot_clusters_2d(self):
        X, y = load_iris()
        pca = PCA(n_components=2)
        X_reduced = pca.fit_transform(X)
        try:
            plot_clusters_2d(X_reduced, y)
        except Exception as e:
            self.fail(f"plot_clusters_2d lanzó una excepción: {e}")


if __name__ == '__main__':
    unittest.main()
