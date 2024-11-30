# clustering_lib/preprocessing/scaling.py

import numpy as np

class StandardScaler:
    """
    Estandariza características eliminando la media y escalando a varianza unitaria.
    """

    def __init__(self):
        self.mean_ = None
        self.scale_ = None

    def fit(self, X):
        X = np.asarray(X)
        self.mean_ = X.mean(axis=0)
        self.scale_ = X.std(axis=0)
        return self

    def transform(self, X):
        X = np.asarray(X)
        return (X - self.mean_) / self.scale_

    def fit_transform(self, X):
        return self.fit(X).transform(X)

    def inverse_transform(self, X_scaled):
        """
        Deshace la transformación y devuelve los datos originales.
        
        Parameters:
            X_scaled (array-like): Datos estandarizados.
        
        Returns:
            X_original (array): Datos originales.
        """
        X_scaled = np.asarray(X_scaled)
        return X_scaled * self.scale_ + self.mean_
