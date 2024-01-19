import os
from tkinter import filedialog

import pandas as pd


def return_dataframe_filesize_filename_from_csv_or_xlsx():
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
