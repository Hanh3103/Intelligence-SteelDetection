import numpy as np
from sklearn.mixture import GaussianMixture
import json
import umap
from sklearn.cluster import AgglomerativeClustering
from sklearn.manifold import TSNE
import skfuzzy as fuzz
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans

'GMM_Algorithmus'


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
           columns_to_cluster (list): List of columns to be used for clustering.
           n_components (int): Number of GMM components.
           random_state (int): Random state for reproducibility.

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


# csv_file_path = 'C:/Users/Mert/OneDrive/Desktop/FoPra/data/cleaned data/combined_OES_data.csv'
# df = pd.read_csv(csv_file_path)
# Beispielaufruf der Funktionen
# df enthält Ihre Daten, columns_to_cluster ist die Liste der zu clusternden Spalten
# columns_to_cluster = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Co', 'Cu', 'Nb', 'Ti', 'V', 'W', 'Pb', 'Sn', 'B', 'Ca', 'Zr', 'As', 'Bi']
# n_components_recommendation = recommend_gmm_components(df[columns_to_cluster])
# result_json = classify_steel_measurements(df, columns_to_cluster, n_components_recommendation)
# Ausgabe des JSON-Formats
# print(result_json)


'Tsne_Algorithmus'


def tsne_algorithm(data, perplexity, mode):
    """
    Applies the t-SNE algorithm to the data.

    Args:
        data (pandas.DataFrame): The data to be transformed.
        perplexity (float): number of comparisons between every datapoint with another datapoint.
        mode (str): The mode of t-SNE, either '2D' or '3D'.

    Returns:
        str: The transformed data in JSON format.
    """
    steps = 10000
    if mode == '2D':
        tsne = TSNE(n_components=2, perplexity=perplexity, n_iter=steps, random_state=0)
        tsne_results_2D = tsne.fit_transform(data)
        tsne_df = pd.DataFrame(tsne_results_2D, columns=['Dimension1', 'Dimension2'])
        return tsne_df.to_json()


    elif mode == '3D':
        tsne = TSNE(n_components=3, perplexity=perplexity, n_iter=steps, random_state=0)
        tsne_results_3D = tsne.fit_transform(data)
        tsne_df = pd.DataFrame(tsne_results_3D, columns=['Dimension1', 'Dimension2'])
        return tsne_df.to_json()


    else:
        print("Invalid mode. Please choose either '2D' or '3D'.")
        return None


def fuzzy_c_mean_fit(data, n_clusters):
    """
    Applies the fuzzy c-means clustering algorithm to the data.

     Args:
        data (pandas.DataFrame): The data to be clustered.
        n_clusters (int): The number of clusters.

        Returns:
            str: The cluster centers and the membership matrix in JSON format.
        """
    try:
        cntr, u, u0, d, jm, p, fpc = fuzz.cluster.cmeans(data.transpose(), n_clusters, 1.4, error=0.005,
                                                         maxiter=10000, init=None)
        result = {
            'cluster_centers': cntr.tolist(),
            'membership_matrix': u.tolist()
        }
        return json.dumps(result)
    except ValueError:
        print("Invalid number of clusters.")
        return None


def fuzzy_c_mean_predict(data, cntr):
    """
    Predicts the membership matrix for the given data using the cluster centers.

    Args:
        data (pandas.DataFrame): The data to be predicted.
        cntr (numpy.ndarray): The cluster centers from fitting.

    Returns:
        str: The membership matrix in JSON format.
    """
    try:
        u, u0, d, jm, p, fpc = fuzz.cluster.cmeans_predict(data.transpose(), cntr, 1.4, error=0.005, maxiter=10000)
        return u.to_json()
    except ValueError:
        print("Invalid cluster centers.")
        return None


'PCA_Algorithmus'


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

    return pca


'K-Means Algorthm'


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


'U-Map'


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

    fit = umap.UMAP(
        n_neighbors=n_neighbors,
        min_dist=min_dist,
        n_components=n_components,
        metric='euclidean'  # Specify the metric for UMAP (e.g., 'euclidean', 'cosine', etc.)
    )
    u = fit.fit_transform(data)
    return u


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

    cluster = AgglomerativeClustering(
        n_clusters=n_clusters,
        affinity='euclidean',  # Specify the distance metric (e.g., 'euclidean', 'cosine', etc.)
        linkage='ward'  # Specify the linkage criterion for clustering (e.g., 'ward', 'complete', 'average', etc.)
    )
    labels = cluster.fit_predict(data)
    data['Cluster'] = labels
    return data
