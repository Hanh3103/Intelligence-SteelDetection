import json
from tkinter import filedialog, Tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import requests

def upload_file():
    # Function to upload a file using tkinter file dialog
    file_path = filedialog.askopenfilename()

    # Extract filename and filesize
    filename = file_path.split('/')[-1]
    filesize = file_path.split('/')[-2]

    # Read the file based on its type (CSV or Excel)
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
    files = {"filename": filename, "filesize": filesize, "data": df_json}
    try:
        header = {"Authorization": f"Bearer {token_save}"}
        print(header)
        response = requests.post(
            "http://127.0.0.1:8000/users/me/item",
            files={"file": (filename, df_json, "application/json")},
            headers=header
        )
        response.raise_for_status()
        print(response.status_code)
    except requests.exceptions.HTTPError as err:
        raise SystemExit(err)


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

    # # Plot the data using Matplotlib
    # plot_frame = Tk()
    # plot_frame.title("Matplotlib Plot in Tkinter")
    #
    # # Assuming 'Date' is a column in your DataFrame
    # x_values = df['Date']
    # # Assuming 'Fe' is another column in your DataFrame
    # y_values = df['Fe']
    #
    # # Create a Matplotlib figure
    # fig, ax = plt.subplots()
    # ax.plot(x_values, y_values, label='Fe over Date')
    # ax.set_xlabel('Date')
    # ax.set_ylabel('Fe')
    # ax.set_title('Matplotlib Plot')
    #
    # # Embed the Matplotlib figure in the Tkinter window
    # canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    # canvas.draw()
    # canvas.get_tk_widget().pack()
    #
    # # Display the Tkinter window
    # plot_frame.mainloop()

    # To be implemented: plot the response data to the gui


# The commented-out section can be uncommented and implemented as needed.
"""
def open_sql_dialog():
    # Function to open a dialog for entering an SQL statement
    sql_string = simpledialog.askstring("Input SQL Statement", "Enter your SQL statement:")
    if sql_string is not None:
        predict_data_from_database(sql_string)

def predict_data_from_database(sql_string):
    # Function to predict data from a database using an SQL statement
    print(f"Processing SQL statement: {sql_string}")
"""


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
