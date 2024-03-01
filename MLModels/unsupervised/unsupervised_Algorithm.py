from cluster_Algorithm import *
from enum import Enum


class models(Enum):
    KMEANS = "KMeans"
    GMM = "GaussianMixture"
    tsne = "Tsne"
    pca = "PCA"
    umap = "Umap"
    hierarchical = "Hierachcial"
    fuzzy = "Fuzzy"


""" 
Paramater:
model: Input algorithm,
df: input dataframe
k_components; number input
scaler: sc_learn
Output Dataframe + Cluster Spalte
"""


def selectAlgorithm(model, df, k_components, scaler=None, ):
    if model == repr(models.GMM):
        return gmm_algorithm(df, k_components)

    if model == repr(models.KMEANS):
        return kmeans(df, k_components, scaler)

    if model == repr(models.hierarchical):
        return hierarchical(df, k_components)


"""
Input
df: input dataframe
scaler: scaler object
Output: PCA
"""


def selectPCA(df, scaler=None):
    return pca(df, scaler)


"""
Input
df: input dataframe
n_neighbors:numbers of neighbors
mode: mode dimension 2d or 3D
Output: TSNE Array
"""


def selectTSNE(df, n_neighbors, mode):
    return tsne_algorithm(df, n_neighbors, mode)


"""
Input
df: input dataframe
n_neighbors:numbers of neighbors
k_components: numbers of components
min_dist: minimum distance between clusters
Output: UMAP dataframe output
"""


def selectUMAP(n_neighbors, k_components, df, min_dist=0.1):
    return apply_umap(n_neighbors, min_dist, k_components, df)


"""
Input
df: input dataframe
k_components: numbers of components
Output: Fuzzy-Matrix
"""


def selectFUZZY(df, k_components):
    return fuzzy_c_mean_fit(df, k_components)
