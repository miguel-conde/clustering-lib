# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# clustering_lib/base.py

from abc import ABC, abstractmethod
import numpy as np


class ClustererBase(ABC):
    """
    Clase base abstracta para todos los algoritmos de clustering.
    """

    @abstractmethod
    def fit(self, X):
        """
        Ajusta el modelo a los datos X.
        
        Parameters:
            X (array-like): Datos de entrada de forma (n_samples, n_features).
        """
        pass

    @abstractmethod
    def predict(self, X):
        """
        Predice las etiquetas para los datos X.
        
        Parameters:
            X (array-like): Datos de entrada de forma (n_samples, n_features).
        
        Returns:
            labels (array): Etiquetas de cluster asignadas a cada muestra.
        """
        pass

    def fit_predict(self, X):
        """
        Combina fit y predict en un solo método.
        
        Parameters:
            X (array-like): Datos de entrada de forma (n_samples, n_features).
        
        Returns:
            labels (array): Etiquetas de cluster asignadas a cada muestra.
        """
        self.fit(X)
        return self.predict(X)
