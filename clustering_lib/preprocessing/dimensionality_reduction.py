# clustering_lib/preprocessing/dimensionality_reduction.py

import numpy as np


class PCA:
    """
    Análisis de Componentes Principales (PCA) para reducción de dimensionalidad.
    """

    def __init__(self, n_components):
        """
        Inicializa el modelo PCA.

        Parameters:
            n_components (int): Número de componentes principales a conservar.
        """
        self.n_components = n_components
        self.components_ = None
        self.mean_ = None

    def fit(self, X):
        """
        Ajusta el modelo PCA a los datos X.

        Parameters:
            X (array-like): Datos de entrada.

        Returns:
            self
        """
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        X_centered = X - self.mean_
        cov_matrix = np.cov(X_centered, rowvar=False)
        eigenvalues, eigenvectors = np.linalg.eigh(cov_matrix)
        # Ordenar los eigenvectores por eigenvalores decrecientes
        idx = np.argsort(eigenvalues)[::-1]
        eigenvectors = eigenvectors[:, idx]
        self.components_ = eigenvectors[:, :self.n_components]
        return self

    def transform(self, X):
        """
        Aplica la transformación PCA a los datos X.

        Parameters:
            X (array-like): Datos a transformar.

        Returns:
            X_reduced (array): Datos transformados.
        """
        X = np.asarray(X)
        X_centered = X - self.mean_
        return np.dot(X_centered, self.components_)

    def fit_transform(self, X):
        """
        Combina fit y transform en un solo método.

        Parameters:
            X (array-like): Datos de entrada.

        Returns:
            X_reduced (array): Datos transformados.
        """
        return self.fit(X).transform(X)
    
    def inverse_transform(self, X_reduced):
        """
        Reconstruye los datos originales a partir de los componentes principales.
        
        Parameters:
            X_reduced (array-like): Datos en el espacio reducido.
        
        Returns:
            X_approx (array): Aproximación de los datos originales.
        """
        return np.dot(X_reduced, self.components_.T) + self.mean_
