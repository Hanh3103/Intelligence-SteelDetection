from enum import Enum


class models(Enum):
    KMEANS = "KMeans"
    GMM = "GaussianMixture"
    tsne = "Tsne"
    pca = "PCA"
    umap = "Umap"
    hierarchical = "Hierachcial"
    fuzzy = "Fuzzy"
