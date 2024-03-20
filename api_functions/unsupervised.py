import io

import pandas as pd
import requests


def predict_data_from_file_unsupervised_kmeans(file, k_components: int, scaler=None):
    """
    Predict data from a file using unsupervised K-Means clustering.

    Args:
        file (str): Path to the file containing the data for prediction.
        k_components (int): Number of clusters for K-Means algorithm.
        scaler (Optional[Scaler]): Scaler to be applied to the data. Defaults to None.

    Returns:
        None.

    Input:
        - `file` (str): Path to the file containing the data for prediction.
        - `k_components` (int): Number of clusters for K-Means algorithm.
        - `scaler` (Optional[Scaler]): Scaler to be applied to the data. Defaults to None.

    Output:
        None. The function prints the reconstructed DataFrame after prediction.
    """
    model_options = {
        "k_components": k_components,
        "scaler": scaler,
    }

    response = call_unsupervised_algorithm(file, "predict/unsupervised/kmeans", model_options)

    # Save response into Object
    json_string = response.json()

    # Use io.StringIO to create a buffer-like object
    json_buffer = io.StringIO(json_string)

    # Read the JSON from the buffer
    df_reconstructed = pd.read_json(json_buffer, orient='split')
    return df_reconstructed


def predict_data_from_file_unsupervised_gmm(file, k_components: int):
    """
    Predict data from a file using unsupervised Gaussian Mixture Model (GMM) clustering.

    Args:
        file (str): Path to the file containing the data for prediction.
        k_components (int): Number of components (clusters) for GMM.

    Returns:
        None.

    Input:
        - `file` (str): Path to the file containing the data for prediction.
        - `k_components` (int): Number of components (clusters) for GMM.

    Output:
        None. The function prints the reconstructed DataFrame after prediction.
    """
    model_options = {
        "k_components": k_components,
    }

    response = call_unsupervised_algorithm(file, "predict/unsupervised/gmm", model_options)

    df_reconstructed = pd.read_json(response.json(), orient='split')
    return df_reconstructed


def predict_data_from_file_unsupervised_hierarchical(file, k_components: int):
    """
    Predict data from a file using unsupervised Hierarchical clustering.

    Args:
        file (str): Path to the file containing the data for prediction.
        k_components (int): Number of clusters for Hierarchical clustering.

    Returns:
        None.

    Input:
        - `file` (str): Path to the file containing the data for prediction.
        - `k_components` (int): Number of clusters for Hierarchical clustering.

    Output:
        None. The function prints the reconstructed DataFrame after prediction.
    """
    model_options = {
        "k_components": k_components,
    }

    response = call_unsupervised_algorithm(file, "predict/unsupervised/hierarchical", model_options)

    df_reconstructed = pd.read_json(response.json(), orient='split')
    return df_reconstructed


def predict_data_from_file_unsupervised_tsne(file, n_neighbours: int, steps: int, mode: str):
    """
    Predict data from a file using unsupervised t-Distributed Stochastic Neighbor Embedding (t-SNE).

    Args:
        file (str): Path to the file containing the data for prediction.
        n_neighbours (int): Number of nearest neighbors for t-SNE algorithm.
        steps (int): Number of optimization steps for t-SNE algorithm.
        mode (str): The mode of t-SNE, either '2D' or '3D'.

    Returns:
        None.

    Input:
        - `file` (str): Path to the file containing the data for prediction.
        - `n_neighbours` (int): Number of nearest neighbors for t-SNE algorithm.
        - `steps` (int): Number of optimization steps for t-SNE algorithm.
        - `mode` (str): Mode of the t-SNE algorithm.

    Output:
        None. The function prints the response JSON and may visualize the response.
    """
    model_options = {
        "n_neighbours": n_neighbours,
        "mode": mode,
        "steps": steps
    }

    response = call_unsupervised_algorithm(file, "predict/unsupervised/tsne", model_options)
    return response.json()


def predict_data_from_file_unsupervised_pca(file, scaler=None):
    """
    Predict data from a file using unsupervised Principal Component Analysis (PCA).

    Args:
        file (str): Path to the file containing the data for prediction.
        scaler (Optional[Scaler]): Scaler to be applied to the data. Defaults to None.

    Returns:
        None.

    Input:
        - `file` (str): Path to the file containing the data for prediction.
        - `scaler` (Optional[Scaler]): Scaler to be applied to the data. Defaults to None.

    Output:
        None. The function prints the response JSON and may visualize the response.
    """
    model_options = {
        "scaler": scaler,
    }

    response = call_unsupervised_algorithm(file, "predict/unsupervised/pca", model_options)
    return response.json()
    # Visualize response


def predict_data_from_file_unsupervised_umap(file, n_neighbours: int, k_components: int, min_dist: float):
    """
    Predict data from a file using unsupervised Uniform Manifold Approximation and Projection (UMAP).

    Args:
        file (str): Path to the file containing the data for prediction.
        n_neighbours (int): Number of nearest neighbors for UMAP.
        k_components (int): Number of components (dimensions) for UMAP.
        min_dist (float): Minimum distance parameter for UMAP.

    Returns:
        None.

    Input:
        - `file` (str): Path to the file containing the data for prediction.
        - `n_neighbours` (int): Number of nearest neighbors for UMAP.
        - `k_components` (int): Number of components (dimensions) for UMAP.
        - `min_dist` (float): Minimum distance parameter for UMAP.

    Output:
        None. The function prints the response JSON and may visualize the response.
    """
    model_options = {
        "n_neighbours": n_neighbours,
        "k_components": k_components,
        "min_dist": min_dist
    }

    response = call_unsupervised_algorithm(file, "predict/unsupervised/umap", model_options)
    return response.json()


def predict_data_from_file_unsupervised_affinity(file):

    response = call_unsupervised_algorithm(file, "predict/unsupervised/affinity")
    df_reconstructed = pd.read_json(response.json(), orient='split')
    return df_reconstructed


def call_unsupervised_algorithm(file, url: str, model_parameter=None):
    """
    Call an unsupervised algorithm API with the specified file, model parameters, and URL.

    Args:
        file (Tuple): A tuple containing DataFrame, filesize, and filename.
        model_parameter (dict): Model parameters for the unsupervised algorithm.
        url (str): URL endpoint for the unsupervised algorithm API.

    Returns:
        requests.Response: The response object from the API call.

    Input:
        - `file` (Tuple): A tuple containing DataFrame, filesize, and filename.
        - `model_parameter` (dict): Model parameters for the unsupervised algorithm.
        - `url` (str): URL endpoint for the unsupervised algorithm API.

    Output:
        requests.Response: The response object from the API call.
    """
    df, filesize, filename = file

    # Convert DataFrame to JSON
    df_json = df.to_json(orient='split')

    data = {
        "model_options": model_parameter,
        "dataframe_input": {
            "data": df_json,
            "filesize": filesize,
            "filename": filename}
    }

    return requests.post("http://127.0.0.1:8000/" + url, json=data)
