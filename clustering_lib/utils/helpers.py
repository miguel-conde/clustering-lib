# clustering_lib/utils/helpers.py

import numpy as np


def validate_input(X):
    """
    Valida los datos de entrada para asegurarse de que son del tipo y forma correctos.

    Parameters:
        X (array-like): Datos de entrada.

    Returns:
        X (ndarray): Datos de entrada convertidos a un array de NumPy.

    Raises:
        ValueError: Si X no es un array 2D.
    """
    X = np.asarray(X)
    if X.ndim != 2:
        raise ValueError("Los datos de entrada deben ser un array 2D.")
    return X


def euclidean_distance(a, b):
    """
    Calcula la distancia euclidiana entre dos puntos o matrices de puntos.

    Parameters:
        a (array-like): Primer punto o matriz de puntos.
        b (array-like): Segundo punto o matriz de puntos.

    Returns:
        distance (float o ndarray): Distancia(s) euclidiana(s) calculada(s).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.linalg.norm(a - b, axis=-1)


def manhattan_distance(a, b):
    """
    Calcula la distancia de Manhattan entre dos puntos o matrices de puntos.

    Parameters:
        a (array-like): Primer punto o matriz de puntos.
        b (array-like): Segundo punto o matriz de puntos.

    Returns:
        distance (float o ndarray): Distancia(s) de Manhattan calculada(s).
    """
    a = np.asarray(a)
    b = np.asarray(b)
    return np.sum(np.abs(a - b), axis=-1)


def cosine_similarity(a, b):
    """
    Calcula la similitud del coseno entre dos vectores.

    Parameters:
        a (array-like): Primer vector.
        b (array-like): Segundo vector.

    Returns:
        similarity (float): Similitud del coseno entre a y b.
    """
    a = np.asarray(a)
    b = np.asarray(b)
    numerator = np.dot(a, b)
    denominator = np.linalg.norm(a) * np.linalg.norm(b)
    if denominator == 0:
        return 0.0
    else:
        return numerator / denominator
