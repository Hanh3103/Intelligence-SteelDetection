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
from MLModels.supervised.supervised_algo_enum import AlgorithmSupervised


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


# ------------------------------------Supervised Algo------------------------------------#
def upload_file_with_file_ending(file_ending):
    file_path = filedialog.askopenfilename()

    filename = file_path.split('/')[-1]

    if file_path.endswith(file_ending):
        with open(file_path, 'rb') as file:
            file_data = file.read()

        return filename, file_data
    elif file_path == '':
        raise Exception("No file selected.")
    else:
        raise Exception("Unsupported file type. Please provide" + file_ending + " file.")


def validate_params(algorithm, table_name_to_predict, existing_model, table_name_to_train):
    # Algorithm and table_name_to_predict should not be empty independently of mode_train
    # Algorithm should not be empty and must be a valid algorithm
    if algorithm == "":
        raise Exception("Please provide an algorithm.")
    elif algorithm not in [algo.value[0] for algo in AlgorithmSupervised]:
        raise Exception("Please provide a valid algorithm.")

    # table_name_to_predict should not be empty
    if table_name_to_predict == "":
        raise Exception("Please provide a table name to predict the data.")

    if existing_model == "False":
        # table_name_to_train should not be empty
        if table_name_to_train == "":
            raise Exception("Please provide a table name to train the model.")
    else:
        # table_name_to_train should be empty
        if table_name_to_train != "":
            raise Exception("Please leave the table name to train the model empty.")


"""
existing_model: True or False (required)
table_name_to_train: name of the table to train the model only needed if existing_model is False
algorithm: name of the algorithm (required)
table_name_to_predict: name of the table to predict the data(required)
name_of_function: Test / Predict/ Save model button (required) > button > maybe redundant 
"""


def upload_encoded_file():
    try:
        # Upload encoded file
        encoded_file_name, encoded_file_data = upload_file_with_file_ending(".pkl")
    except ValueError as e:
        encoded_file_name = ""
        encoded_file_data = b''
    except Exception as e:
        print(e)
        return
    return encoded_file_name, encoded_file_data


def upload_model_file(algorithm):
    try:
        model_file_name = ""
        model_file_data = b''
        # Upload model file
        if algorithm == AlgorithmSupervised.DeepLearning.value[0]:
            model_file_name, model_file_data = upload_file_with_file_ending(AlgorithmSupervised.DeepLearning.value[1])
        elif algorithm == AlgorithmSupervised.Xgboost.value[0]:
            model_file_name, model_file_data = upload_file_with_file_ending(AlgorithmSupervised.Xgboost.value[1])
        return model_file_name, model_file_data
    except ValueError as e:
        print(e)
        return
    except Exception as e:
        print(e)
        return


def predict_with_supervised_algo(existing_model, table_name_to_train, algorithm, table_name_to_predict,
                                 name_of_function):
    # Validate params
    try:
        validate_params(algorithm, table_name_to_predict, existing_model, table_name_to_train)
    except Exception as e:
        print(e)
        return

    # Upload files
    if existing_model == "False":  # train new model and predict
        # upload encoded file > is optional
        encoded_file_name, encoded_file_data = upload_encoded_file()

        files = {"upload_model_file": ("", b''),
                 "encoded_file": (encoded_file_name, encoded_file_data)}

    else:  # use upload existing model and predict
        # Upload model file, ending of model is depend on the algorithm > is  required
        model_file_name, model_file_data = upload_model_file(algorithm)
        # Upload encoded file > is optional
        encoded_file_name, encoded_file_data = upload_encoded_file()

        # Include the uploaded files in the request payload
        files = {"upload_model_file": (model_file_name, model_file_data),
                 "encoded_file": (encoded_file_name, encoded_file_data)}

    # Include parameters in the request payload
    payload = {
        "existing_model": existing_model,
        "table_name_to_train": table_name_to_train,  # this is the table name of the training data, the table muss first
        # be uploaded over "Data import"
        "algorithm": algorithm,
        "table_name_to_predict": table_name_to_predict,  # this is the table name of the data to predict, the table muss
        # first be uploaded over "Data import"
        "name_of_function": name_of_function  # Test / Predict/ Save model button
    }
    payload = json.dumps(payload)

    data = {
        "up_model_options": payload
    }

    header = {"accept": "application/json", "Authorization": f"Bearer {token_save}"}
    response = requests.post("http://127.0.0.1:8000/predict/supervised", files=files,
                             data=data, headers=header)

    if response.status_code != 200:
        print(response.json().get("detail"))

    # Process the response data
    res = response.json()
    prediction = res.get("prediction")
    print(prediction)


# ------------------------------------End Supervised------------------------------------#


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

        print(dict_dataframes.keys())

        show_dataframes(dict_dataframes)

    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def show_dataframes(dict_dataframes):  # show dataframes in gui but not elegant!
    show(**dict_dataframes)


def delete_item_by_name(name):
    try:
        header = {"Authorization": f"Bearer {token_save}"}
        response = requests.delete(f"http://127.0.0.1:8000/users/me/item/{name}", headers=header)

        if response.status_code == 401:
            print(response.json().get("detail"))
        else:
            print(f"Item with name {name} deleted.")

    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)
