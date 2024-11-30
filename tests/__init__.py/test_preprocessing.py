# tests/test_preprocessing.py

import unittest
import numpy as np
from clustering_lib.preprocessing.scaling import StandardScaler


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


if __name__ == '__main__':
    unittest.main()
