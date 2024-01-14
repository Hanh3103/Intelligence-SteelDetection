import json
import os
from io import StringIO
from tkinter import filedialog, Tk
from typing import Optional

from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import requests
from pandasgui import show
from MLModels.unsupervised import cluster_Algorithm

from enums import unsupervised_models_enum


def upload_file():
    file_path = filedialog.askopenfilename()

    filename = file_path.split('/')[-1]
    filesize = os.path.getsize(file_path)

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file type. Please provide a CSV or Excel file.")
        return None

    return df, filesize, filename


def only_upload_file():
    df, filesize, filename = upload_file()
    # Convert DataFrame to JSON
    df_json = df.to_json(orient='split')

    # Include the DataFrame in the request payload
    data = {"data_frame": {"filename": filename, "filesize": filesize, "data": df_json}}
    try:
        header = {"Authorization": f"Bearer {token_save}"}
        response = requests.post("http://127.0.0.1:8000/users/me/item", json=data, headers=header)

        if response.status_code == 401:
            print(response.json().get("detail"))

    except requests.exceptions.HTTPError as err:
        print(err)
    except requests.exceptions.ConnectionError as err:
        print(err)


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


# Aktuell wird standardmäßig kmeans algorithmus aufgerufen bei unsupervised button click
# Kann man später lösen indem man hier ein switch case einbaut oder aus der gui gleich
# den jeweiligen Algorithmus aufruft.
# Testweise kann man hier auch die anderen unsupervised algorithmen aufrufen,
# diese sind ebenfalls weiter oben implementiert.
# Als Uploaddatei bitte test_data_unsupervised.csv aus dem Ordner /resources übergeben,
# da nur für diese Datei ein Output garantiert ist

def predict_data_from_file_unsupervised():
    # Function to predict data from an uploaded file in unsupervised mode
    # df, filesize, filename = upload_file()

    predict_data_from_file_unsupervised_kmeans(upload_file(), 15, None)


def predict_data_from_file_supervised():
    # Function to predict data from an uploaded file in supervised mode
    df, filesize, filename = upload_file()

    # TODO only take the first 10 rows from the dataframe > only temporary
    df = df.head(10)

    # TODO remove the last column from the dataframe > only temporary
    df = df.iloc[:, :-1]

    # Convert DataFrame to JSON
    df_json = df.to_json(orient='split')

    # Include the DataFrame in the request payload
    files = {"dataframe_input": {"data": df_json, "filesize": filesize, "filename": filename}}

    # write files to a json file for testing with swagger later
    with open('files.json', 'w') as outfile:
        json.dump(files, outfile)

    response = requests.post("http://127.0.0.1:8000/predict/supervised", json=files)

    # Process the response data
    data = response.json()
    prediction = data.get("prediction")
    print(prediction)


def register_user(username, password, full_name, email):
    username = username
    password = password
    full_name = full_name
    email = email
    json_data = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "password": password,
        "disabled": False
    }

    try:
        response = requests.post("http://127.0.0.1:8000/registrate", json=json_data)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def login_user(username, password):
    username = username
    password = password

    try:
        response = requests.post("http://127.0.0.1:8000/token", data={"username": username, "password": password})
        response.raise_for_status()

        response_json = response.json()
        token = response_json["access_token"]
        token_t = response_json["token_type"]
        global token_save
        global token_type
        token_save = token
        token_type = token_t
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def get_all_items():
    try:
        header = {"Authorization": f"Bearer {token_save}"}
        response = requests.get("http://127.0.0.1:8000/users/me/items", headers=header)

        if response.status_code == 401:
            print(response.json().get("detail"))

        list_json = json.loads(response.content)

        # turn into list of dataframes
        dict_dataframes = {}
        for item in list_json:
            name = item["name"]
            data = item["data"]
            df = pd.read_json(StringIO(data), orient='split')
            dict_dataframes[name] = df

        show_dataframes(dict_dataframes)

    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def show_dataframes(dict_dataframes):  # show dataframes in gui but not elegant!
    show(**dict_dataframes)
