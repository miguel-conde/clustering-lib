# tests/test_algorithms.py

import unittest
import numpy as np
from clustering_lib.algorithms.kmeans import KMeansClusterer
from clustering_lib.datasets.load_datasets import load_iris


class TestKMeansClusterer(unittest.TestCase):
    def setUp(self):
        self.X, self.y = load_iris()

    def test_kmeans_fit_predict(self):
        kmeans = KMeansClusterer(n_clusters=3, random_state=42)
        labels = kmeans.fit_predict(self.X)
        self.assertEqual(len(labels), len(self.X))
        self.assertEqual(len(set(labels)), 3)

    def test_kmeans_convergence(self):
        kmeans = KMeansClusterer(n_clusters=3, max_iter=1, random_state=42)
        kmeans.fit(self.X)
        initial_inertia = kmeans.inertia_
        kmeans.max_iter = 100
        kmeans.fit(self.X)
        final_inertia = kmeans.inertia_
        self.assertLessEqual(final_inertia, initial_inertia)

    def test_kmeans_predict(self):
        kmeans = KMeansClusterer(n_clusters=3, random_state=42)
        kmeans.fit(self.X)
        new_points = np.array([[5.0, 3.5, 1.5, 0.2],
                               [6.5, 3.0, 5.5, 2.0]])
        labels = kmeans.predict(new_points)
        self.assertEqual(len(labels), len(new_points))


if __name__ == '__main__':
    unittest.main()
