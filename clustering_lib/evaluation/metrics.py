# -*- coding: utf-8 -*-
#
# MIT License
#
# Copyright (c) 2024 Miguel Conde
#
# clustering_lib/evaluation/metrics.py

import numpy as np


def silhouette_score(X, labels):
    """
    Calcula el coeficiente de silueta promedio de todas las muestras.

    Parameters:
        X (array-like): Datos de entrada.
        labels (array-like): Etiquetas de cluster asignadas a las muestras.

    Returns:
        score (float): Coeficiente de silueta promedio.
    """
    from sklearn.metrics import silhouette_score as sk_silhouette_score
    return sk_silhouette_score(X, labels)
