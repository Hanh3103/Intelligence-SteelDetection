import pandas as pd
from fastapi import APIRouter

from MLModels.unsupervised import cluster_Algorithm
from utils.unsupervised_utils import extract_model_options_and_df
from utils.unsupervised_utils import convert_json_to_df

router = APIRouter()


@router.post("/predict/unsupervised/kmeans")
async def predict_unsupervised_kmeans(data: dict):
    """
    Endpoint for predicting unsupervised clustering using the K-Means algorithm.

    Args:
        data (dict): Dictionary containing model options and DataFrame for prediction.

    Returns:
        str: JSON representation of the clustered data.

    Notes:
        Utilizes the 'cluster_Algorithm.kmeans' method for prediction.
    """
    extracted_data = await extract_model_options_and_df(data)

    return cluster_Algorithm.kmeans(extracted_data["data_frame"], extracted_data["model_options"]["k_components"],
                                    extracted_data["model_options"]["scaler"]).to_json(orient="split")


@router.post("/predict/unsupervised/gmm")
async def predict_unsupervised_gmm(data: dict):
    """
    Endpoint for predicting unsupervised clustering using the Gaussian Mixture Model (GMM) algorithm.

    Args:
        data (dict): Dictionary containing model options and DataFrame for prediction.

    Returns:
        str: JSON representation of the clustered data.

    Notes:
        Utilizes the 'cluster_Algorithm.gmm_algorithm' method for prediction.
    """
    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.gmm_algorithm(extracted_data["data_frame"],
                                                       extracted_data["model_options"]["k_components"])

    if isinstance(response_predict, pd.DataFrame):
        return response_predict.to_json(orient='split')
    else:
        return response_predict


@router.post("/predict/unsupervised/hierarchical")
async def predict_unsupervised_hierarchical(data: dict):
    """
    Endpoint for predicting unsupervised clustering using the Hierarchical clustering algorithm.

    Args:
        data (dict): Dictionary containing model options and DataFrame for prediction.

    Returns:
        str: JSON representation of the clustered data.

    Notes:
        Utilizes the 'cluster_Algorithm.hierarchical' method for prediction.
    """
    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.hierarchical(extracted_data["data_frame"],
                                                      extracted_data["model_options"]["k_components"])
    return response_predict.to_json(orient='split')


@router.post("/predict/unsupervised/tsne")
async def predict_unsupervised_tsne(data: dict):
    """
    Endpoint for predicting unsupervised clustering using the t-Distributed Stochastic Neighbor Embedding (t-SNE)
    algorithm.

    Args:
        data (dict): Dictionary containing model options and DataFrame for prediction.

    Returns:
        str: JSON representation of the clustered data.

    Notes:
        Utilizes the 'cluster_Algorithm.tsne_algorithm' method for prediction.
    """
    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.tsne_algorithm(extracted_data["data_frame"],
                                                        extracted_data["model_options"]["n_neighbours"],
                                                        extracted_data["model_options"]["steps"],
                                                        extracted_data["model_options"]["mode"])
    return response_predict


@router.post("/predict/unsupervised/pca")
async def predict_unsupervised_pca(data: dict):
    """
    Endpoint for predicting unsupervised clustering using Principal Component Analysis (PCA).

    Args:
        data (dict): Dictionary containing model options and DataFrame for prediction.

    Returns:
        list: List representation of the clustered data.

    Notes:
        Utilizes the 'cluster_Algorithm.pca' method for prediction.
    """
    extracted_data = await extract_model_options_and_df(data)

    response_pca = cluster_Algorithm.pca(extracted_data["data_frame"], extracted_data["model_options"]["scaler"])
    return response_pca.tolist()


@router.post("/predict/unsupervised/umap")
async def predict_unsupervised_umap(data: dict):
    """
    Endpoint for predicting unsupervised clustering using Uniform Manifold Approximation and Projection (UMAP).

    Args:
        data (dict): Dictionary containing model options and DataFrame for prediction.

    Returns:
        list: List representation of the clustered data.

    Notes:
        Utilizes the 'cluster_Algorithm.apply_umap' method for prediction.
    """
    extracted_data = await extract_model_options_and_df(data)

    response_data = cluster_Algorithm.apply_umap(extracted_data["data_frame"],
                                                 extracted_data["model_options"]["n_neighbours"],
                                                 extracted_data["model_options"]["min_dist"],
                                                 extracted_data["model_options"]["k_components"])
    return response_data.tolist()


@router.post("/predict/unsupervised/affinity")
async def predict_unsupervised_affinity(data: dict):

    df_input = data.get("dataframe_input").get("data")
    df = await convert_json_to_df(df_input)

    return cluster_Algorithm.affinity_propagation(df).to_json(orient="split")
