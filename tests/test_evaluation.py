# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# tests/test_evaluation.py

import unittest
import numpy as np
from clustering_lib.evaluation.metrics import silhouette_score
from clustering_lib.algorithms.kmeans import KMeansClusterer
from clustering_lib.datasets.load_datasets import load_iris


class TestEvaluationMetrics(unittest.TestCase):
    def test_silhouette_score(self):
        X, y = load_iris()
        kmeans = KMeansClusterer(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        self.assertTrue(-1 <= score <= 1)
        self.assertGreater(score, 0.0)


if __name__ == '__main__':
    unittest.main()
