import numpy as np
from sklearn.mixture import GaussianMixture
import json
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
from sklearn.cluster import AffinityPropagation
from sklearn.metrics import pairwise_distances
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.preprocessing import FunctionTransformer


# GMM Algorithmus
def recommend_gmm_components(data, max_components=10):
    """
      Recommends the number of Gaussian Mixture Model (GMM) components based on the Bayesian Information Criterion (BIC).

      Parameters:
          data (pd.DataFrame): Input dataset.
          max_components (int): Maximum number of GMM components to be tested.

      Returns:
          int: Recommended number of GMM components.
      """
    random_state = 0
    bic_values = []
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(data)
        bic_values.append(gmm.bic(data))

    # Find the number of components with the lowest BIC value
    recommended_components = np.argmin(bic_values) + 1
    return recommended_components


def gmm_algorithm(data, n_components):
    """
       Classifies steel measurements using the Gaussian Mixture Model (GMM) algorithm and returns the results in JSON format.

       Parameters:
           data (pd.DataFrame): Input dataset.
           n_components (int): Number of GMM components.

       Returns:
           str: JSON format of the GMM results.
       """
    # Extract relevant data
    random_state = 0
    numeric_cols = data.select_dtypes(include=['number']).columns
    data_to_cluster = data[numeric_cols]

    # Create and fit GMM model
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    clusters_gmm = gmm.fit_predict(data_to_cluster)

    # Add GMM clusters to the data
    data['Cluster'] = clusters_gmm

    return data


# TSNE Algorithmus
def tsne_algorithm(data, perplexity, steps, mode):
    """
     Applies the t-SNE algorithm to the data.

     Args:
         data (pandas.DataFrame): The data to be transformed.
         perplexity (float): number of comparisons between every datapoint with another datapoint.
         steps (int): The number of iterations for t-SNE.
         mode (str): The mode of t-SNE, either '2D' or '3D'.

     Returns:
         str: The transformed data in JSON format.
     """

    columns_to_scale = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Co', 'Cu', 'Nb', 'Ti', 'V', 'W', 'Pb',
                        'Sn', 'B', 'Ca', 'Zr', 'As', 'Bi']

    scaler = FunctionTransformer(np.log1p, validate=True)
    scaled_data = scaler.fit_transform(data[columns_to_scale])

    df_scaled = pd.DataFrame(scaled_data, columns=columns_to_scale)

    if mode == '2D':
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=steps, random_state=0)
        tsne_results_2D = tsne.fit_transform(df_scaled)
        return tsne_results_2D.tolist()

    elif mode == '3D':
        tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=steps, random_state=0)
        tsne_results_3D = tsne.fit_transform(df_scaled)
        return tsne_results_3D.tolist()

    else:
        print("Invalid mode. Please choose either '2D' or '3D'.")
        return None
    

# PCA Algorithmus
def pca(dataframe, scaler=None):
    """
    Apply optional scaling and PCA to the numeric columns of a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to process.
    scaler (optional): An instance of a Scaler from sklearn.preprocessing. Default is None.

    Returns:
    PCA: The PCA model fitted to the scaled numeric columns of the DataFrame.
    """
    # Check if the DataFrame is valid
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame.")

    # Select numeric columns
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns.")

    numeric_data = dataframe[numeric_cols]

    # Apply scaling if a scaler is provided
    if scaler is not None:
        numeric_data = scaler.fit_transform(numeric_data)

    # Apply PCA
    pca = PCA()
    pca.fit(numeric_data)

    return pca.explained_variance_ratio_


# KMEANS Algorithmus
def kmeans(dataframe, k, scaler=None):
    """
    Apply k-means clustering to the DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to process.
    k (int): Number of clusters.
    scaler (optional): An instance of a Scaler from sklearn.preprocessing. Default is None.

    Returns:
    pd.DataFrame: Original DataFrame with an additional column 'Cluster' indicating the cluster assignment.

    Warnings:
        Throws UserWarning on Windows as KMeans is known to have a memory leak on Windows.
    """
    # Check if the DataFrame is valid
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame.")

    # Check if the DataFrame is empty
    if dataframe.empty:
        raise ValueError("Input DataFrame is empty.")

    # Validate k
    if not isinstance(k, int) or k <= 1:
        raise ValueError("k must be an integer greater than 1.")

    # Select numeric columns
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns.")

    numeric_data = dataframe[numeric_cols]

    # Apply scaling if a scaler is provided
    if scaler is not None:
        numeric_data = scaler.fit_transform(numeric_data)

    # Apply k-means clustering
    kmeans_c = KMeans(n_clusters=k, random_state=0, n_init=10)
    dataframe['Cluster'] = kmeans_c.fit_predict(numeric_data)

    return dataframe


# UMAP Algorithmus
def apply_umap(data, n_neighbors=15, min_dist=0.1, n_components=2):
    """
    Applies Uniform Manifold Approximation and Projection (UMAP) to reduce the dimensionality of the input data.

    Parameters:
        - data: Input data to be transformed (must be a pandas DataFrame).
        - n_neighbors: The size of local neighborhood used for manifold approximation (default is 15).
        - min_dist: The minimum distance between points in the low-dimensional representation (default is 0.1).
        - n_components: The number of dimensions in the embedded space (default is 2).

    Returns:
        - u: The low-dimensional representation of the input data.
    """

    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")

    numeric_cols = data.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns.")

    numeric_data = data[numeric_cols]

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean'  # Specify the metric for UMAP (e.g., 'euclidean', 'cosine', etc.)
    )
    u = fit.fit_transform(numeric_data)
    return u

# Hierarchical Algorithmus
def hierarchical(data, n_clusters):
    """
        Applies Hierarchical Agglomerative Clustering to the input data.

        Parameters:
            - data: Input data for clustering (must be a pandas DataFrame).
            - n_clusters: The number of clusters to form.

        Returns:
            - data: The input data with an additional 'Cluster' column indicating the assigned cluster for each data point.
        """
    if not isinstance(data, pd.DataFrame):
        raise ValueError("Input data must be a pandas DataFrame.")
    numeric_cols = data.select_dtypes(include=['number']).columns
    data_to_cluster = data[numeric_cols]
    cluster = AgglomerativeClustering(n_clusters, metric='euclidean', linkage='ward')
    cluster.fit(data_to_cluster)
    labels = cluster.labels_
    data['Cluster'] = labels
    return data


def affinity_propagation(data):
    """
        Applies Affinity Propagation Clustering to the input data.

        Parameters:
            - data: Input data for clustering (must be a pandas DataFrame).

        Returns:
            - data: The input data with an additional 'Cluster' column indicating the assigned cluster for each data point.
        """
    #apply pca for better results
    pca = PCA(n_components=3) 
    df_pca = pca.fit_transform(data)

    #apply affinity propagation
    algorithm = AffinityPropagation(damping=0.5, preference=np.min(-pairwise_distances(df_pca, metric='euclidean')))
    algorithm.fit(df_pca)
    labels = algorithm.labels_
    data['Cluster'] = labels
    return data

