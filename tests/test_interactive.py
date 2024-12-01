# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# tests/test_visualization.py

import unittest
from clustering_lib.visualization.interactive import interactive_clustering
from clustering_lib.datasets.load_datasets import load_iris
import matplotlib

# Usar backend 'Agg' para evitar mostrar gráficos durante las pruebas
matplotlib.use('Agg')

class TestInteractive(unittest.TestCase):

    def setUp(self):
        X, y = load_iris()
        self.X = X
        self.y = y

    def test_interactive_clustering(self):
        try:
            # No podemos probar interactividad, pero verificamos que la función se ejecuta sin errores
            interactive_clustering(self.X, max_k=5)
        except Exception as e:
            self.fail(f"interactive_clustering lanzó una excepción: {e}")


if __name__ == '__main__':
    unittest.main()
