# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# tests/test_preprocessing.py

import unittest
import numpy as np
from clustering_lib.preprocessing.scaling import StandardScaler
from clustering_lib.preprocessing.dimensionality_reduction import PCA


class TestStandardScaler(unittest.TestCase):
    def test_fit_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        mean = X_scaled.mean(axis=0)
        std = X_scaled.std(axis=0)
        np.testing.assert_array_almost_equal(mean, np.array([0.0, 0.0]))
        np.testing.assert_array_almost_equal(std, np.array([1.0, 1.0]))

    def test_inverse_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        X_inv = scaler.inverse_transform(X_scaled)
        np.testing.assert_array_almost_equal(X, X_inv)

class TestPCA(unittest.TestCase):

    def test_pca_fit_transform(self):
        X = np.array([[1, 2], [3, 4], [5, 6]])
        pca = PCA(n_components=1)
        X_reduced = pca.fit_transform(X)
        self.assertEqual(X_reduced.shape[1], 1)

    def test_pca_inverse_transform(self):
        X = np.random.rand(10, 5)
        pca = PCA(n_components=3)
        X_reduced = pca.fit_transform(X)
        X_approx = pca.inverse_transform(X_reduced)
        # La reconstrucción no será exacta, pero podemos verificar que las formas coinciden
        self.assertEqual(X_approx.shape, X.shape)
        # Opcionalmente, comprobar que el error de reconstrucción es razonable
        reconstruction_error = np.mean((X - X_approx) ** 2)
        self.assertLess(reconstruction_error, 0.1)


if __name__ == '__main__':
    unittest.main()
