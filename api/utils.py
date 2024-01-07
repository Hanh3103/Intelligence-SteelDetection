import json
import os
from io import StringIO
from tkinter import filedialog, Tk, Label, Frame

import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from pandastable import Table
from pandasgui import show


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


def register_user():
    username = "hanh3"
    full_name = "hanhNguyen"
    email = "test3@te.de"
    password = "123456"
    disabled = False
    json_data = {
        "username": username,
        "full_name": full_name,
        "email": email,
        "password": password,
        "disabled": disabled
    }

    try:
        response = requests.post("http://127.0.0.1:8000/registrate", json=json_data)
        response.raise_for_status()
        print(response.json())
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def login_user():
    username = "hanh3"
    password = "123456"

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


def predict_data_from_file_unsupervised():
    df, filesize, filename = upload_file()

    df_json = df.to_json(orient='split')

    # Include the DataFrame in the request payload
    files = {"dataframe_input": {"data": df_json, "filesize": filesize, "filename": filename}}
    response = requests.post("http://127.0.0.1:8000/predict/unsupervised", json=files)

    data = response.json()
    df2 = pd.DataFrame.from_dict(data)
    print(df2)

    # To be implemented: plot the response data to gui


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
    # To be implemented: plot the response data to gui


def get_all_items():
    try:
        header = {"Authorization": f"Bearer {token_save}"}
        response = requests.get("http://127.0.0.1:8000/users/me/items", headers=header)

        if response.status_code == 401:
            print(response.json().get("detail"))

        list_items = json.loads(response.content)

        # turn into list of dataframes
        dict_dataframes = {}
        for name, table_json in list_items.items():
            table = pd.read_json(StringIO(table_json), orient='split')
            dict_dataframes[name] = table

        show_dataframes(dict_dataframes)

    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def show_dataframes(dict_dataframes):  # show dataframes in gui but not elegant!
    show(**dict_dataframes)
