# clustering_lib/evaluation/selection.py

import numpy as np
import matplotlib.pyplot as plt
from clustering_lib.algorithms.kmeans import KMeansClusterer
from clustering_lib.evaluation.metrics import silhouette_score
from sklearn.metrics import adjusted_rand_score
from sklearn.model_selection import StratifiedKFold

def plot_elbow_method(X, max_k=10):
    """
    Genera un gráfico del método del codo para ayudar a seleccionar el número óptimo de clusters.

    Parameters:
        X (array-like): Datos de entrada.
        max_k (int): Número máximo de clusters a probar.
    """
    inertias = []
    Ks = range(1, max_k + 1)
    for k in Ks:
        kmeans = KMeansClusterer(n_clusters=k)
        kmeans.fit(X)
        inertias.append(kmeans.inertia_)

    plt.figure(figsize=(8, 4))
    plt.plot(Ks, inertias, 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Inercia')
    plt.title('Método del Codo para Determinar k Óptimo')
    plt.xticks(Ks)
    plt.grid(True)
    plt.show()

def plot_silhouette_scores(X, max_k=10):
    """
    Genera un gráfico del coeficiente de silueta promedio para diferentes valores de k.

    Parameters:
        X (array-like): Datos de entrada.
        max_k (int): Número máximo de clusters a probar.
    """
    silhouette_scores = []
    Ks = range(2, max_k + 1)  # Silhouette score no está definido para k=1

    for k in Ks:
        kmeans = KMeansClusterer(n_clusters=k)
        labels = kmeans.fit_predict(X)
        score = silhouette_score(X, labels)
        silhouette_scores.append(score)

    plt.figure(figsize=(8, 4))
    plt.plot(Ks, silhouette_scores, 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Coeficiente de Silueta Promedio')
    plt.title('Análisis del Coeficiente de Silueta para Determinar k Óptimo')
    plt.xticks(Ks)
    plt.grid(True)
    plt.show()

def gap_statistic(X, max_k=10, n_references=10, random_state=None):
    """
    Calcula la estadística Gap para determinar el número óptimo de clusters.

    Parameters:
        X (array-like): Datos de entrada.
        max_k (int): Número máximo de clusters a probar.
        n_references (int): Número de conjuntos de referencia aleatorios.
        random_state (int): Semilla para el generador de números aleatorios.

    Returns:
        optimal_k (int): Número óptimo de clusters.
        gaps (list): Valores de la estadística Gap para cada k.
    """
    from sklearn.cluster import KMeans
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)
    shape = X.shape
    gaps = []
    deviations = []
    ks = range(1, max_k + 1)

    # Crear n conjuntos de referencia aleatorios
    reference_inertia = np.zeros((len(ks), n_references))

    for i in range(n_references):
        # Datos aleatorios uniformemente distribuidos dentro del espacio de los datos
        X_ref = rng.uniform(np.min(X, axis=0), np.max(X, axis=0), size=shape)

        for idx, k in enumerate(ks):
            kmeans = KMeans(n_clusters=k, random_state=random_state)
            kmeans.fit(X_ref)
            reference_inertia[idx, i] = kmeans.inertia_

    # Inercia de los datos originales
    ondata_inertia = []
    for k in ks:
        kmeans = KMeans(n_clusters=k, random_state=random_state)
        kmeans.fit(X)
        ondata_inertia.append(kmeans.inertia_)

    # Calcular la estadística Gap
    log_ref_inertia = np.mean(np.log(reference_inertia), axis=1)
    log_ondata_inertia = np.log(ondata_inertia)
    gaps = log_ref_inertia - log_ondata_inertia

    # Determinar el k óptimo
    optimal_k = ks[np.argmax(gaps)]

    # Graficar la estadística Gap
    plt.figure(figsize=(8, 4))
    plt.plot(ks, gaps, 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Estadística Gap')
    plt.title('Estadística Gap para Determinar k Óptimo')
    plt.xticks(ks)
    plt.grid(True)
    plt.show()

    return optimal_k, gaps


def cluster_stability_analysis(X, max_k=10, n_splits=5, random_state=None):
    """
    Evalúa la estabilidad de los clusters mediante validación cruzada interna.

    Parameters:
        X (array-like): Datos de entrada.
        max_k (int): Número máximo de clusters a probar.
        n_splits (int): Número de particiones para la validación cruzada.
        random_state (int): Semilla para el generador de números aleatorios.

    Returns:
        stability_scores (dict): Diccionario con k como clave y puntaje de estabilidad como valor.
    """
    from sklearn.cluster import KMeans
    from sklearn.utils import check_random_state

    rng = check_random_state(random_state)
    stability_scores = {}
    ks = range(2, max_k + 1)

    for k in ks:
        scores = []
        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=rng)
        for train_index, test_index in skf.split(X, np.zeros(X.shape[0])):
            X_train, X_test = X[train_index], X[test_index]

            # Ajustar el modelo en el conjunto de entrenamiento
            kmeans_train = KMeans(n_clusters=k, random_state=rng)
            kmeans_train.fit(X_train)
            labels_train = kmeans_train.predict(X_train)

            # Predecir en el conjunto de prueba
            labels_test = kmeans_train.predict(X_test)

            # Ajustar el modelo en el conjunto de prueba
            kmeans_test = KMeans(n_clusters=k, random_state=rng)
            kmeans_test.fit(X_test)
            labels_test_2 = kmeans_test.labels_

            # Calcular el puntaje de estabilidad
            score = adjusted_rand_score(labels_test, labels_test_2)
            scores.append(score)

        stability_scores[k] = np.mean(scores)

    # Graficar los puntajes de estabilidad
    plt.figure(figsize=(8, 4))
    plt.plot(list(stability_scores.keys()), list(stability_scores.values()), 'bo-')
    plt.xlabel('Número de clusters (k)')
    plt.ylabel('Puntaje de Estabilidad (Adjusted Rand Index)')
    plt.title('Análisis de Estabilidad de Clusters')
    plt.xticks(ks)
    plt.grid(True)
    plt.show()

    return stability_scores

def plot_combined_metrics(X, max_k=10):
    """
    Grafica múltiples métricas (inercia, coeficiente de silueta) en un solo gráfico.

    Parameters:
        X (array-like): Datos de entrada.
        max_k (int): Número máximo de clusters a probar.
    """
    inertias = []
    silhouette_scores = []
    Ks = range(2, max_k + 1)

    for k in Ks:
        kmeans = KMeansClusterer(n_clusters=k)
        labels = kmeans.fit_predict(X)
        inertias.append(kmeans.inertia_)
        silhouette_scores.append(silhouette_score(X, labels))

    fig, ax1 = plt.subplots(figsize=(8, 4))

    color = 'tab:red'
    ax1.set_xlabel('Número de clusters (k)')
    ax1.set_ylabel('Inercia', color=color)
    ax1.plot(Ks, inertias, 'o-', color=color, label='Inercia')
    ax1.tick_params(axis='y', labelcolor=color)
    ax1.grid(True)

    ax2 = ax1.twinx()

    color = 'tab:blue'
    ax2.set_ylabel('Coeficiente de Silueta', color=color)
    ax2.plot(Ks, silhouette_scores, 's-', color=color, label='Coeficiente de Silueta')
    ax2.tick_params(axis='y', labelcolor=color)

    plt.title('Comparación de Métricas para Determinar k Óptimo')
    fig.tight_layout()
    plt.show()
