from cluster_Algorithm import *
from enum import Enum
class modelsClustering(Enum):
    KMEANS = "KMeans"
    GMM = "GaussianMixture"
    hierarchical = "Hierarchical"
    fuzzy        = "Fuzzy"

class modelsReduction(Enum):
    tsne = "Tsne"
    umap = "Umap"
    pca = "PCA"

""" 
Paramater:
model: Input algorithm,
df: input dataframe
k_components; number input
scaler: sc_learn
Output Dataframe + Cluster Spalte
"""
def selectClusteringAlgorithm(model,df,k_components,scaler=None,):
    print(model)
    if model == modelsClustering.GMM.value:
       return gmm_algorithm(df,k_components)

    elif model == modelsClustering.KMEANS.value:
       return kmeans(df,k_components,scaler)

    elif model == modelsClustering.hierarchical.value:
       return hierarchical(df,k_components)
    else:
        raise ValueError("Invalid clustering model. Supported models: KMeans, GaussianMixture, Hierarchical, Fuzzy KMeans ")


def selectReductionAlgorithm(df, reduction_model, **kwargs):
    """
        Selects and applies the specified dimensionality reduction algorithm on the input dataframe.

        Parameters:
            - df: Input dataframe.
            - reduction_model: An instance of the ModelsReduction enum indicating the desired reduction algorithm.
            - **kwargs: Additional parameters specific to the chosen algorithm.

        Returns:
            - Output of the dimensionality reduction algorithm.
        """
    if not isinstance(df, pd.DataFrame):
        raise ValueError("Input dataframe must be a pandas DataFrame.")

    if reduction_model == modelsReduction.tsne.value:
        if 'perplexity' not in kwargs or 'mode' not in kwargs:
            raise ValueError("For TSNE, 'n_neighbors' and 'mode' are required arguments.")
        return tsne_algorithm(df, **kwargs)
    elif reduction_model == modelsReduction.umap.value:
        if 'n_neighbors' not in kwargs or 'n_components' not in kwargs or 'min_dist' not in kwargs:
            raise ValueError("For UMAP, 'n_neighbors' and 'k_components' are required arguments.")
        return apply_umap(df, **kwargs)
    elif reduction_model == modelsReduction.pca.value:
        return pca(df, **kwargs)
    else:
        raise ValueError("Invalid reduction model. Supported models: Tsne, Umap, PCA")

"""
Input
df: input dataframe
scaler: scaler object
Output: PCA
"""
def selectPCA(df,scaler=None):
    return pca(df,scaler)
"""
Input
df: input dataframe
n_neighbors:numbers of neighbors
mode: mode dimension 2d or 3D
Output: TSNE Array
"""
def selectTSNE(df,n_neighbors,mode):
    return tsne_algorithm(df,n_neighbors,mode)

"""
Input
df: input dataframe
n_neighbors:numbers of neighbors
k_components: numbers of components
min_dist: minimum distance between clusters
Output: UMAP dataframe output
"""
def selectUMAP(n_neighbors,k_components,df,min_dist=0.1):
    return apply_umap(n_neighbors,min_dist,k_components,df)

"""
Input
df: input dataframe
k_components: numbers of components
Output: Fuzzy-Matrix
"""
def selectFUZZY(df,k_components):
    return fuzzy_c_mean_fit(df,k_components)
