import tkinter as tk
from io import BytesIO
from tkinter import filedialog
import pandas as pd
import requests
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from scipy.optimize import root
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


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

    # To be implemented: plot the response data to gui



"""
def open_sql_dialog():
    sql_string = simpledialog.askstring("Input SQL Statement", "Enter your SQL statement:")
    if sql_string is not None:
        predict_data_from_database(sql_string)


def predict_data_from_database(sql_string):
    print(f"Processing SQL statement: {sql_string}")
"""