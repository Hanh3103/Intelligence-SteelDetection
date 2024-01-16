import pandas as pd
from fastapi import APIRouter

from MLModels.unsupervised import cluster_Algorithm
from utils.unsupervised_utils import extract_model_options_and_df

router = APIRouter()


@router.post("/predict/unsupervised/kmeans")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    return cluster_Algorithm.kmeans(extracted_data["data_frame"], extracted_data["model_options"]["k_components"],
                                    extracted_data["model_options"]["scaler"]).to_json(orient="split")


@router.post("/predict/unsupervised/gmm")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.gmm_algorithm(extracted_data["data_frame"],
                                                       extracted_data["model_options"]["k_components"])

    if isinstance(response_predict, pd.DataFrame):
        return response_predict.to_json(orient='split')
    else:
        return response_predict


@router.post("/predict/unsupervised/hierarchical")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.hierarchical(extracted_data["data_frame"],
                                                      extracted_data["model_options"]["k_components"])
    return response_predict.to_json(orient='split')


@router.post("/predict/unsupervised/tsne")
async def predict_unsupervised_tsne(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.tsne_algorithm(extracted_data["data_frame"],
                                                        extracted_data["model_options"]["n_neighbours"],
                                                        extracted_data["model_options"]["steps"],
                                                        extracted_data["model_options"]["mode"])
    return response_predict


@router.post("/predict/unsupervised/pca")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_pca = cluster_Algorithm.pca(extracted_data["data_frame"], extracted_data["model_options"]["scaler"])
    return response_pca.tolist()


@router.post("/predict/unsupervised/umap")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_data = cluster_Algorithm.apply_umap(extracted_data["data_frame"],
                                                 extracted_data["model_options"]["n_neighbours"],
                                                 extracted_data["model_options"]["min_dist"],
                                                 extracted_data["model_options"]["k_components"])
    return response_data.tolist()
