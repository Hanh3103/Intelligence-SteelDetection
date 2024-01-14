import json
import os
from io import StringIO
from tkinter import filedialog, Tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import requests
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
    # Function to predict data from an uploaded file in unsupervised mode
    df, filesize, filename = upload_file()

    # Convert DataFrame to JSON
    df_json = df.to_json(orient='split')

    # Include the DataFrame in the request payload
    files = {"dataframe_input": {"data": df_json, "filesize": filesize, "filename": filename}}
    response = requests.post("http://127.0.0.1:8000/predict_supervised", json=files)

    # Process the response data
    data = response.json()
    df2 = pd.DataFrame.from_dict(data)
    print(df2)

    # Plot the data using Matplotlib
    plot_frame = Tk()
    plot_frame.title("Matplotlib Plot in Tkinter")

    # Assuming 'Date' is a column in your DataFrame
    x_values = df['Date']
    # Assuming 'Fe' is another column in your DataFrame
    y_values = df['Fe']

    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='Fe over Date')
    ax.set_xlabel('Date')
    ax.set_ylabel('Fe')
    ax.set_title('Matplotlib Plot')

    # Embed the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Display the Tkinter window
    plot_frame.mainloop()

    # To be implemented: plot the response data to the gui


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


def upload_transformed_dataframe_to_steel_db():
    steel_data = return_transformed_dataframe_to_steel_data()

    steel_data_json = steel_data.to_json(orient='split')
    header = {"Authorization": f"Bearer {token_save}"}
    json_data = {
        'filename': steel_data['strFileName'].iloc[0],
        'steeldata': steel_data_json
    }

    response = requests.post("http://127.0.0.1:8000/users/me/upload_steel_data", headers=header, json=json_data)


def return_transformed_dataframe_to_steel_data():
    # Step 1: Get DataFrame from CSV or Excel file
    dataframe, filename = return_dataframe_and_filename_from_csv_or_xlsx()

    # Step 2: Extract columns, indices, and raw_data directly from DataFrame
    columns = dataframe.columns
    raw_data = dataframe.values.tolist()

    transformed_data = []

    # Step 3: Iterate over raw data to transform and append to the new structure
    for idx, row in enumerate(raw_data):
        steel_type = row[-1]  # Assuming the steel class is the last column
        for i, value in enumerate(row[:-1]):
            material = columns[i]
            material_value = float(str(value).replace(',', '.'))  # Convert comma to dot and parse as float
            transformed_data.append({
                'intSteelDataID': idx,
                'strFileName': filename,
                'strSteelType': steel_type,
                'strMaterial': material,
                'doubleMaterialValue': material_value
            })

    # Step 4: Convert transformed data to Pandas DataFrame
    df_transformed_data = pd.DataFrame(transformed_data)

    return df_transformed_data


def return_dataframe_and_filename_from_csv_or_xlsx():
    file_path = filedialog.askopenfilename()
    filename = os.path.basename(file_path)

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file type. Please provide a CSV or Excel file.")
        return None

    return df, filename
