import json
from io import StringIO

import pandas as pd
import requests

from api_functions.login_register import get_token
from utils.database_utils import show_dataframes
from utils.upload_csv_or_xlsx import return_dataframe_filesize_filename_from_csv_or_xlsx


def upload_csv_or_xlsx_to_db_table():
    """
    Upload a CSV or XLSX file to a database table by sending a POST request.

    Raises:
        - HTTPError: If an HTTP error occurs during the request.
        - ConnectionError: If a connection error occurs during the request.

    Returns:
        None.

    Input: - The function uses the global function `get_token()` to retrieve the access token. - The function
    internally calls `return_dataframe_filesize_filename_from_csv_or_xlsx()` to get the DataFrame, filesize,
    and filename.

    Output:
        None. The function prints error details if a 401 status code is encountered.
    """
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
        # Print HTTP error details
        print(err)
    except requests.exceptions.ConnectionError as err:
        # Print connection error details
        print(err)


def delete_db_table_by_name(name):
    """
    Delete a database table by sending a DELETE request with the specified name.

    Args:
        name (str): The name of the database table to be deleted.

    Raises:
        SystemExit: If an HTTP error occurs during the request.

    Returns:
        None.

    Input:
        - `name` (str): The name of the database table to be deleted. This is provided as a function parameter.

    Output:
        None. The function prints error details if a 401 status code is encountered or a success message otherwise.
    """
    token_save = get_token()
    try:
        header = {"Authorization": f"Bearer {token_save}"}
        response = requests.delete(f"http://127.0.0.1:8000/users/me/item/{name}", headers=header)

        if response.status_code == 401:
            # Print error details if a 401 status code is encountered
            print(response.json().get("detail"))
        else:
            # Print a success message if the deletion is successful
            print(f"Item with name {name} deleted.")

    except requests.exceptions.HTTPError as err:
        # Exit the system if an HTTP error occurs
        raise SystemExit(err)


def get_all_items_of_user_from_db():
    """
    Retrieve all items of the user from the database by sending a GET request.

    Raises:
        SystemExit: If an HTTP error occurs during the request.

    Returns:
        A dictionary containing dataframes associated with each item retrieved from the database.
    """
    token_save = get_token()
    try:
        header = {"Authorization": f"Bearer {token_save}"}
        response = requests.get("http://127.0.0.1:8000/users/me/items", headers=header)

        if response.status_code != 200:
            # Print error details if the response status code is not 200
            print(response.json().get("detail"))
            # You might want to handle this error case more gracefully based on your application's requirements
            return {}

        list_json = json.loads(response.content)

        # Check if there's any data returned
        if not list_json:
            print("No data available in the database.")
            return {}

        # Turn data into a dictionary of dataframes
        dict_dataframes = {}
        for item in list_json:
            name = item["name"]
            data = item["data"]
            df = pd.read_json(StringIO(data), orient='split')
            dict_dataframes[name] = df

        # Print the names of items retrieved
        print(dict_dataframes.keys())

        # Return the dictionary containing dataframes
        return dict_dataframes

    except requests.exceptions.HTTPError as err:
        # Exit the system if an HTTP error occurs
        raise SystemExit(err)
