# clustering_lib/datasets/load_datasets.py

from sklearn import datasets


def load_iris():
    """
    Carga el conjunto de datos Iris.

    Returns:
        X (array): Características de las muestras.
        y (array): Etiquetas reales de las muestras.
    """
    iris = datasets.load_iris()
    return iris.data, iris.target
