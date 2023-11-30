import pandas as pd
import glob
import os


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
        if "chemie" not in file.lower():
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
    combined_df = pd.concat(data_frames.values(), axis=0, ignore_index=True, sort=False)
    # Ersetze alle NaN-Werte mit 0 im kombinierten DataFrame
    combined_df = combined_df.fillna(0)

    # Konvertiere float 0 und 1 zu int 0 und 1 im kombinierten DataFrame
    def convert_floats(val):
        if val == 0.0 or val == 1.0:
            return int(val)
        return val

    combined_df = combined_df.applymap(convert_floats)

    return combined_df


# TODO: remove or use
def drop_zero_burn_rows(df):
    """ Entfernt alle Zeilen, in denen die Spalte 'Burn' den Wert 0 hat. """
    return df[df['Burn'] != 0]


# Pfad zum Ordner setzen
folder_path = 'data/OES/'
processed_data = process_excel_files(folder_path)
combined_df = merge_data_frames(processed_data)
combined_df = combined_df.drop_duplicates()

# TODO: umgang mit nicht trivialen dublikaten
# Liste der Spalten, auf denen die Duplikatsuche basieren soll
columns_to_check = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Co',
                    'Cu', 'Nb', 'Ti', 'V', 'W', 'Pb', 'Sn', 'B', 'Ca', 'Zr', 'As', 'Bi']

# Finde Duplikate im DataFrame
duplicate_rows = combined_df[combined_df.duplicated(subset=columns_to_check, keep='first')]

# Anzeigen der gefundenen Duplikate
print(duplicate_rows)

# speichern des DataFrames als csv
combined_df.to_csv('data/cleaned data/combined_OESdata.csv', index=False)

# TODO: decision how to handle column 'Grade' with 3 types of values (0, typ of steel, grade location?); POSSIBLE SOLUTION. merge with column ID1
# TODO: decision how to handle columns ID1,2,3; unnamed; columns with greater/lesser but few values (eg Ni_lesser, Ni_greater)
