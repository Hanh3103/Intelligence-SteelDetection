import json
from io import StringIO

import pandas as pd
import requests

from api_functions.login_register import get_token
from utils.database_utils import show_dataframes
from utils.upload_csv_or_xlsx import return_dataframe_filesize_filename_from_csv_or_xlsx


def upload_csv_or_xlsx_to_db_table():
    token_save = get_token()
    df, filesize, filename = return_dataframe_filesize_filename_from_csv_or_xlsx()
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


def delete_db_table_by_name(name):
    token_save = get_token()
    try:
        header = {"Authorization": f"Bearer {token_save}"}
        response = requests.delete(f"http://127.0.0.1:8000/users/me/item/{name}", headers=header)

        if response.status_code == 401:
            print(response.json().get("detail"))
        else:
            print(f"Item with name {name} deleted.")

    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


def get_all_items_of_user_from_db():
    token_save = get_token()
    try:
        header = {"Authorization": f"Bearer {token_save}"}
        response = requests.get("http://127.0.0.1:8000/users/me/items", headers=header)

        if response.status_code != 200:
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
