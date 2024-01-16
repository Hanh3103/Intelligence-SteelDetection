import pandas as pd
import requests


def predict_data_from_file_unsupervised_kmeans(file, k_components: int, scaler=None):
    model_options = {
        "k_components": k_components,
        "scaler": scaler,
    }

    response = call_unsupervised_algorithm(file, model_options, "predict/unsupervised/kmeans")

    df_reconstructed = pd.read_json(response.json(), orient='split')
    print(df_reconstructed)
    # Visualize response


def predict_data_from_file_unsupervised_gmm(file, k_components: int):
    model_options = {
        "k_components": k_components,
    }

    response = call_unsupervised_algorithm(file, model_options, "predict/unsupervised/gmm")

    df_reconstructed = pd.read_json(response.json(), orient='split')
    print(df_reconstructed)
    # Visualize response


def predict_data_from_file_unsupervised_hierarchical(file, k_components: int):
    model_options = {
        "k_components": k_components,
    }

    response = call_unsupervised_algorithm(file, model_options, "predict/unsupervised/hierarchical")

    df_reconstructed = pd.read_json(response.json(), orient='split')
    print(df_reconstructed)
    # Visualize response


def predict_data_from_file_unsupervised_tsne(file, n_neighbours: int, steps: int, mode: str):
    model_options = {
        "n_neighbours": n_neighbours,
        "mode": mode,
        "steps": steps
    }

    response = call_unsupervised_algorithm(file, model_options, "predict/unsupervised/tsne")
    print(response.json())
    # Visualize response


def predict_data_from_file_unsupervised_pca(file, scaler=None):
    model_options = {
        "scaler": scaler,
    }

    response = call_unsupervised_algorithm(file, model_options, "predict/unsupervised/pca")
    print(response.json())
    # df_reconstructed = pd.read_json(response.json(), orient='split')
    # print(df_reconstructed)
    # Visualize response


def predict_data_from_file_unsupervised_umap(file, n_neighbours: int, k_components: int, min_dist: float):
    model_options = {
        "n_neighbours": n_neighbours,
        "k_components": k_components,
        "min_dist": min_dist
    }

    response = call_unsupervised_algorithm(file, model_options, "predict/unsupervised/umap")
    print(response.json())
    # Visualize response


def call_unsupervised_algorithm(file, model_parameter, url: str):
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