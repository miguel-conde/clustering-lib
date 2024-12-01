# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# tests/test_evaluation.py

import unittest
import numpy as np
from clustering_lib.evaluation.selection import (
    plot_elbow_method,
    plot_silhouette_scores,
    gap_statistic,
    cluster_stability_analysis,
    plot_combined_metrics,
)
from clustering_lib.datasets.load_datasets import load_iris

# Usar backend 'Agg' para evitar mostrar gráficos durante las pruebas
import matplotlib
matplotlib.use('Agg')


class TestEvaluationMethods(unittest.TestCase):

    def setUp(self):
        X, y = load_iris()
        self.X = X
        self.y = y

    def test_plot_elbow_method(self):
        try:
            plot_elbow_method(self.X, max_k=5)
        except Exception as e:
            self.fail(f"plot_elbow_method lanzó una excepción: {e}")

    def test_plot_silhouette_scores(self):
        try:
            plot_silhouette_scores(self.X, max_k=5)
        except Exception as e:
            self.fail(f"plot_silhouette_scores lanzó una excepción: {e}")

    def test_gap_statistic(self):
        try:
            optimal_k, gaps = gap_statistic(self.X, max_k=5, n_references=5, random_state=42)
            self.assertTrue(1 <= optimal_k <= 5)
        except Exception as e:
            self.fail(f"gap_statistic lanzó una excepción: {e}")

    def test_cluster_stability_analysis(self):
        try:
            stability_scores = cluster_stability_analysis(self.X, max_k=5, n_splits=3, random_state=42)
            self.assertTrue(len(stability_scores) > 0)
        except Exception as e:
            self.fail(f"cluster_stability_analysis lanzó una excepción: {e}")

    def test_plot_combined_metrics(self):
        try:
            plot_combined_metrics(self.X, max_k=5)
        except Exception as e:
            self.fail(f"plot_combined_metrics lanzó una excepción: {e}")


if __name__ == '__main__':
    unittest.main()
