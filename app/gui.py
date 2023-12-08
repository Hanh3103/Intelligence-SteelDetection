import tkinter as tk
from tkinter import filedialog, simpledialog
import requests


def open_file_dialog():
    file_path = filedialog.askopenfilename()
    predict_data_from_file(file_path)


def predict_data_from_file(file_path):
    files = {"file": open(file_path, "rb")}
    response = requests.post("http://127.0.0.1:8000/predict", files=files)
    prediction_result = response.json()
    result_label.config(text=f"Prediction: {prediction_result['prediction']}")


def open_sql_dialog():
    sql_string = simpledialog.askstring("Input SQL Statement", "Enter your SQL statement:")
    if sql_string is not None:
        predict_data_from_database(sql_string)


def predict_data_from_database(sql_string):
    print(f"Processing SQL statement: {sql_string}")


# Tkinter GUI setup
root = tk.Tk()
root.title("Machine Learning App")

open_button = tk.Button(root, text="Open File", command=open_file_dialog)
open_button.pack(pady=10)

open_sql_button = tk.Button(root, text="Open SQL Dialog", command=open_sql_dialog)
open_sql_button.pack(pady=10)

result_label = tk.Label(root, text="Prediction: ")
result_label.pack(pady=10)

root.mainloop()
