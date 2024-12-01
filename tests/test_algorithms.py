# tests/test_algorithms.py

import unittest
import numpy as np
from clustering_lib.algorithms.kmeans import KMeansClusterer
from clustering_lib.algorithms.hierarchical import HierarchicalClusterer
from clustering_lib.algorithms.dbscan import DBSCANClusterer
from clustering_lib.datasets.load_datasets import load_iris
from clustering_lib.preprocessing.scaling import StandardScaler


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

class TestHierarchicalClusterer(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris()
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def test_hierarchical_fit_predict(self):
        hierarchical = HierarchicalClusterer(n_clusters=3, linkage_method='ward')
        labels = hierarchical.fit_predict(self.X_scaled)
        self.assertEqual(len(labels), len(self.X_scaled))
        self.assertEqual(len(set(labels)), 3)

    def test_hierarchical_invalid_predict(self):
        hierarchical = HierarchicalClusterer()
        hierarchical.fit(self.X_scaled)
        with self.assertRaises(NotImplementedError):
            hierarchical.predict(self.X_scaled)


class TestDBSCANClusterer(unittest.TestCase):

    def setUp(self):
        self.X, self.y = load_iris()
        scaler = StandardScaler()
        self.X_scaled = scaler.fit_transform(self.X)

    def test_dbscan_fit_predict(self):
        dbscan = DBSCANClusterer(eps=0.5, min_samples=5)
        labels = dbscan.fit_predict(self.X_scaled)
        self.assertEqual(len(labels), len(self.X_scaled))
        # Verificar que hay al menos un cluster (excluyendo ruido)
        self.assertTrue(len(set(labels)) > 1)

    def test_dbscan_invalid_predict(self):
        dbscan = DBSCANClusterer()
        dbscan.fit(self.X_scaled)
        with self.assertRaises(NotImplementedError):
            dbscan.predict(self.X_scaled)


if __name__ == '__main__':
    unittest.main()
