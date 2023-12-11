import json
from tkinter import filedialog
import pandas as pd
import requests
from io import StringIO
from fastapi.responses import JSONResponse
import plotly.express as px


def upload_file():
    file_path = filedialog.askopenfilename()

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file type. Please provide a CSV or Excel file.")
        return None

    return df


def predict_data_from_file_unsupervised():
    df = upload_file()

    if df is None:
        return  # Exit function if file upload fails

    df_json = df.to_json(orient='split')

    files = {"dataframe_input": {"data": df_json}}
    response = requests.post("http://127.0.0.1:8000/predict_unsupervised", json=files)

    # To be implemented: plot the response data to gui


def predict_data_from_file_supervised():
    df = upload_file()

    df_json = df.to_json(orient='split')

    # Include the DataFrame in the request payload
    files = {"dataframe_input": {"data": df_json}}
    response = requests.post("http://127.0.0.1:8000/predict_supervised", json=files)

    dict = json.loads(response.json())

    df = pd.DataFrame.from_dict(dict)

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

    # To be implemented: plot the response data to gui


"""
def open_sql_dialog():
    sql_string = simpledialog.askstring("Input SQL Statement", "Enter your SQL statement:")
    if sql_string is not None:
        predict_data_from_database(sql_string)


def predict_data_from_database(sql_string):
    print(f"Processing SQL statement: {sql_string}")
"""
