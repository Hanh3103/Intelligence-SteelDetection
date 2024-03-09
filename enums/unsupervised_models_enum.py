from enum import Enum


class Models(Enum):
    """
    Enum class for Machine Learning Models.

    Each enum member represents a machine learning model with a unique identifier.

    Enum Members:
        - KMEANS: Represents the K-Means clustering model.
        - GMM: Represents the Gaussian Mixture Model.
        - tsne: Represents the t-SNE (t-distributed Stochastic Neighbor Embedding) model.
        - pca: Represents the Principal Component Analysis (PCA) model.
        - umap: Represents the Uniform Manifold Approximation and Projection (UMAP) model.
        - hierarchical: Represents the Hierarchical clustering model.
        - fuzzy: Represents the Fuzzy clustering model.
    """

    KMEANS = "KMeans"
    GMM = "GaussianMixture"
    tsne = "Tsne"
    pca = "PCA"
    umap = "Umap"
    hierarchical = "Hierarchical"
    affinity = "affinity "

    @staticmethod
    def create_list():
        return [model.value for model in Models]

    @staticmethod
    def get_enum_by_value(value):
        for model in Models:
            if model.value == value:
                return model
        return None
