# clustering_lib/visualization/plots.py

import matplotlib.pyplot as plt


def plot_clusters_2d(X, labels):
    """
    Grafica los clusters en 2D.

    Parameters:
        X (array-like): Datos de entrada de forma (n_samples, 2).
        labels (array-like): Etiquetas de cluster asignadas a las muestras.
    """
    plt.scatter(X[:, 0], X[:, 1], c=labels, cmap='viridis', alpha=0.7)
    plt.xlabel('Componente 1')
    plt.ylabel('Componente 2')
    plt.title('Clusters')
    plt.show()
