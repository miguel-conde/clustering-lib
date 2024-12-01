# Diagrama de Clases de la Biblioteca de Clustering

```mermaid
classDiagram
    %% Clases Base y Abstractas
    class ClustererBase {
        <<abstract>>
        +fit(X)
        +predict(X)
        +fit_predict(X)
        -labels_
        -n_clusters_
    }

    %% Algoritmos de Clustering
    class KMeansClusterer {
        -n_clusters
        -max_iter
        -tol
        -random_state
        -cluster_centers_
        -inertia_
        +fit(X)
        +predict(X)
        +fit_predict(X)
    }

    class HierarchicalClusterer {
        -n_clusters
        -linkage_method
        -affinity
        -linkage_matrix_
        +fit(X)
        +fit_predict(X)
        +predict(X) $NotImplemented
    }

    class DBSCANClusterer {
        -eps
        -min_samples
        -metric
        -core_sample_indices_
        +fit(X)
        +fit_predict(X)
        +predict(X) $NotImplemented
    }

    %% Preprocesamiento
    class StandardScaler {
        -mean_
        -scale_
        +fit(X)
        +transform(X)
        +fit_transform(X)
        +inverse_transform(X_scaled)
    }

    class PCA {
        -n_components
        -components_
        -mean_
        +fit(X)
        +transform(X)
        +fit_transform(X)
        +inverse_transform(X_reduced)
    }

    %% Evaluación
    class Metrics {
        +silhouette_score(X, labels)
        +calinski_harabasz_score(X, labels)
        +davies_bouldin_score(X, labels)
    }

    %% Visualización
    class Visualization {
        +plot_clusters_2d(X, labels)
        +plot_clusters_3d(X, labels)
        +plot_dendrogram(model)
    }

    %% Utilidades
    class Helpers {
        +validate_input(X)
        +euclidean_distance(a, b)
        +manhattan_distance(a, b)
        +cosine_similarity(a, b)
    }

    %% Relaciones de Herencia
    ClustererBase <|-- KMeansClusterer
    ClustererBase <|-- HierarchicalClusterer
    ClustererBase <|-- DBSCANClusterer

    %% Relaciones de Uso
    KMeansClusterer ..> Helpers : utiliza
    HierarchicalClusterer ..> Helpers : utiliza
    DBSCANClusterer ..> Helpers : utiliza
    StandardScaler ..> Helpers : utiliza
    PCA ..> Helpers : utiliza

    %% Relaciones de Dependencia
    KMeansClusterer ..> Metrics : evalúa con
    HierarchicalClusterer ..> Metrics : evalúa con
    KMeansClusterer ..> Visualization : visualiza con
    HierarchicalClusterer ..> Visualization : visualiza con
    PCA ..> Visualization : visualiza con

    %% Notas
    note for ClustererBase "Clase abstracta que define la interfaz común para algoritmos de clustering."
    note for HierarchicalClusterer::predict "No implementado: Clustering jerárquico no soporta predicciones."
    note for DBSCANClusterer::predict "No implementado: DBSCAN no soporta predicciones."
```
