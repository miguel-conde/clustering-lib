# tests/test_utils.py

import unittest
import numpy as np
from clustering_lib.utils.helpers import (
    validate_input,
    euclidean_distance,
    manhattan_distance,
    cosine_similarity,
)


class TestUtils(unittest.TestCase):

    def test_validate_input(self):
        # Prueba con una entrada válida
        X = np.array([[1, 2], [3, 4]])
        X_validated = validate_input(X)
        np.testing.assert_array_equal(X, X_validated)

        # Prueba con una entrada inválida (no 2D)
        X_invalid = [1, 2, 3]
        with self.assertRaises(ValueError):
            validate_input(X_invalid)

    def test_euclidean_distance(self):
        a = np.array([0, 0])
        b = np.array([3, 4])
        distance = euclidean_distance(a, b)
        self.assertEqual(distance, 5.0)

        # Prueba con matrices
        a = np.array([[0, 0], [1, 1]])
        b = np.array([[3, 4], [4, 5]])
        distances = euclidean_distance(a, b)
        np.testing.assert_array_almost_equal(distances, [5.0, 5.0])

    def test_manhattan_distance(self):
        a = np.array([1, 2])
        b = np.array([4, 6])
        distance = manhattan_distance(a, b)
        self.assertEqual(distance, 7)

    def test_cosine_similarity(self):
        a = np.array([1, 0, 0])
        b = np.array([0, 1, 0])
        similarity = cosine_similarity(a, b)
        self.assertEqual(similarity, 0.0)

        # Prueba con vectores idénticos
        similarity = cosine_similarity(a, a)
        self.assertEqual(similarity, 1.0)


if __name__ == '__main__':
    unittest.main()
