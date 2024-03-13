import pandas as pd
import glob
import os
import numpy as np

columns_to_extract = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Co',
 'Cu', 'Nb', 'Ti', 'V', 'W', 'Pb', 'Sn', 'B', 'Ca', 'Zr', 'As', 'Bi']

def one_hot_encode_column(df, column):
    """ One-hot encode the specified column. """
    df[column + '_lesser'] = df[column].apply(lambda x: 1 if x == '<' else 0)
    df[column + '_greater'] = df[column].apply(lambda x: 1 if x == '>' else 0)
    return df.drop(column, axis=1)

def preprocess_sheet(df):
    """ Preprocess a single sheet. """
    # Umwandeln der ersten Spalte in Datumsangaben, nicht-Datumsangaben werden zu NaT
    df[df.columns[0]] = pd.to_datetime(df[df.columns[0]], errors='coerce')

    # Entfernen von Zeilen, in denen der Wert in der ersten Spalte NaT ist
    df = df[df[df.columns[0]].notna()]

    # Entfernen leerer Zeilen und Spalten
    df = df.dropna(how='all').dropna(how='all', axis=1)

    # One-Hot-Encoding für Spalten mit <, >
    for column in df.columns:
        if df[column].dtype == object and any(df[column].astype(str).str.contains('<|>', regex=True)):
            df = one_hot_encode_column(df, column)

    # Entfernen leerer Spalten, die nur NaN oder leere Strings enthalten
    df = df.drop(columns=[col for col in df.columns if
                          df[col].isnull().all() or all(df[col].astype(str).apply(str.strip) == '')])
    return df

def remove_suffix(df):
    """ Entfernt das Suffix '.1' von Spaltennamen. """
    df.columns = [col.split('.1')[0] if '.1' in col else col for col in df.columns]
    return df

def process_excel_files(folder_path):
    """ Process all Excel files in the specified folder. """
    data_frames = {}
    for file in glob.glob(os.path.join(folder_path, '*.xlsx')):
        if "chemie" not in file.lower() and "oes_ostbahnhof_rohdaten" not in file.lower() and "kopie" not in file.lower():
            xls = pd.ExcelFile(file)
            for sheet_name in xls.sheet_names:
                df = preprocess_sheet(xls.parse(sheet_name))
                # Entferne das Suffix '.1' von Spaltennamen, falls vorhanden
                df = remove_suffix(df)
                key = f"{os.path.basename(file)}_{sheet_name}"
                data_frames[key] = df
    return data_frames

def merge_data_frames(data_frames):
    """ Vereinige alle DataFrames in einem gemeinsamen DataFrame. """
    selected_columns_list = []

    # Iterate through the dictionary and extract the specified columns
    for key, df in data_frames.items():
        selected_columns = df[columns_to_extract]
        selected_columns_list.append(selected_columns)

    # Concatenate the selected columns row-wise into a new DataFrame
    combined_df = pd.concat(selected_columns_list, ignore_index=True)
    # Ersetze alle NaN-Werte mit 0 im kombinierten DataFrame
    combined_df = combined_df.fillna(0)

    # Konvertiere float 0 und 1 zu int 0 und 1 im kombinierten DataFrame
    def convert_floats(val):
        if val == 0.0 or val == 1.0:
            return int(val)
        return val
    combined_df = combined_df.applymap(convert_floats)
    return combined_df


names = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo'
        ,'Ni', 'Al', 'Co', 'Cu', 'Nb', 'Ti', 'V', 'W', 'Pb', 'Sn', 'B', 'Zr', 'As', 'Bi']


def handleOstBahnhof():
    df = pd.read_excel("data/OES/OES_Ostbahnhof_Rohdaten_Final_211021.xlsx")
    df = df.iloc[:, 22:]
    filter = [col for col in df.columns if len(col) >= 3]
    df = df[filter]
    df = df.dropna(how='all').dropna(how='all', axis=1)
    df.columns = names
    ## writing
    df.to_csv('data/cleaned data/Ostbahnhof_cleaned.csv', index=False)
    return df


def handleKopieMesstage():
    names = ['C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Co', 'Cu',
       'Nb', 'Ti', 'V', 'W', 'Pb', 'Sn', 'B', 'Ca', 'Zr', 'As', 'Bi']
    dataframes = {}
    xls = pd.ExcelFile("data/OES/Kopie von P2207_Messtage21-2709_Nachuntersuchung.xlsx")
    for sheet_name in xls.sheet_names:
        df = xls.parse(sheet_name).iloc[:, 7:]
        filter = [col for col in df.columns if len(col) >= 3]
        df = df[filter]
        df = df.dropna(how='all').dropna(how='all', axis=1)
        df.columns = names
        dataframes[sheet_name] = df
    result = pd.concat(dataframes.values(), ignore_index=True)
    result.to_csv('data/cleaned data/KopieMesstage_cleaned.csv', index=False)



def runCleaning():
    # Pfad zum Ordner setzen
    folder_path = 'data/OES/'
    processed_data = process_excel_files(folder_path)
    combined_df = merge_data_frames(processed_data)
    combined_df = combined_df.drop_duplicates()
    # speichern des DataFrames als csv
    combined_df.to_csv('data/cleaned data/combined_OESdata_1.csv', index=False)
    handleOstBahnhof()
    handleKopieMesstage()


runCleaning()
