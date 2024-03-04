import os
from tkinter import filedialog
import pandas as pd
from fastapi import HTTPException, status

def return_dataframe_filesize_filename_from_csv_or_xlsx():
    """
    Opens a file dialog to choose a CSV or Excel file, reads it into a DataFrame,
    and returns the DataFrame, file size, and filename.

    Returns:
    - Tuple containing a DataFrame, file size, and filename.
    """
    try:
        filetypes = [
            ('CSV and Excel files', '*.csv;*.xlsx')]
        file_path = filedialog.askopenfilename( title='Choose Dataframe',filetypes=filetypes)

        if not file_path:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="No file selected.")

        filename = os.path.basename(file_path)
        filesize = os.path.getsize(file_path)

        if file_path.endswith('.csv'):
            df = pd.read_csv(file_path)
        elif file_path.endswith(('.xls', '.xlsx')):
            df = pd.read_excel(file_path)
        else:
            raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Unsupported file type. Please "
                                                                                "provide a CSV or Excel file.")

        return df, filesize, filename
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error processing file: {str(e)}")
