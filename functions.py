import pandas as pd
from sklearn.decomposition import PCA
from sklearn.cluster import KMeans


def process_dataframe(input_df):
    """
    Process the input DataFrame and return a DataFrame.

    Parameters:
    input_df (pd.DataFrame): A pandas DataFrame to be processed.

    Returns:
    pd.DataFrame: The processed DataFrame.
    """

    # Validate input
    if not isinstance(input_df, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame.")

    # return Error if input is empty
    if input_df.empty:
        return input_df

    # return Error if input doesn't contain any numeric columns
    if not any(input_df.select_dtypes(include=['number']).columns):
        raise ValueError("Input DataFrame does not contain any numeric columns.")

    # return Error if inpout doesn't contain column 'Grade'
    if 'Grade' not in input_df.columns:
        raise ValueError("Input DataFrame does not contain column 'Grade'.")

    # Ensure 'Date' column is recognized as datetime
    if 'Date' in input_df.columns:
        try:
            input_df['Date'] = pd.to_datetime(input_df['Date'], errors='coerce')
        except Exception as e:
            print(f"Warning: Error converting 'Date' column to datetime. {e}")

    # Drop columns that contain '<' or '>' in any row
    cols_to_drop = input_df.apply(lambda col: any(col.astype(str).str.contains('<|>', regex=True)))
    input_df = input_df.drop(columns=input_df.columns[cols_to_drop])

    # Convert all columns except 'Date' and 'Grade' to numeric
    cols_to_convert = input_df.columns.drop(['Date', 'Grade'])
    input_df[cols_to_convert] = input_df[cols_to_convert].apply(pd.to_numeric, errors='coerce')

    # Remove empty rows and columns
    input_df.dropna(how='all', axis=0, inplace=True)  # Remove rows with all NaN values
    input_df.dropna(how='all', axis=1, inplace=True)  # Remove columns with all NaN values

    # Remove duplicates
    input_df.drop_duplicates(inplace=True)

    # Group by 'Grade' and calculate mean of the aggregated rows
    if 'Grade' in input_df.columns:
        output_df = input_df.groupby('Grade').mean(numeric_only=True)
    else:
        output_df = input_df
        print("Warning: Column 'Grade' not found. No grouping and mean calculation was performed.")

    # Reset index
    output_df.reset_index(inplace=True)

    # Return the processed DataFrame
    return output_df


def apply_pca_with_scaling(dataframe, scaler=None):
    """
    Apply optional scaling and PCA to the numeric columns of a DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to process.
    scaler (optional): An instance of a Scaler from sklearn.preprocessing. Default is None.

    Returns:
    PCA: The PCA model fitted to the scaled numeric columns of the DataFrame.
    """
    # Check if the DataFrame is valid
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame.")

    # Select numeric columns
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns.")

    numeric_data = dataframe[numeric_cols]

    # Apply scaling if a scaler is provided
    if scaler is not None:
        numeric_data = scaler.fit_transform(numeric_data)

    # Apply PCA
    pca = PCA()
    pca.fit(numeric_data)

    return pca


def apply_kmeans_clustering(dataframe, k, scaler=None):
    """
    Apply k-means clustering to the DataFrame.

    Parameters:
    dataframe (pd.DataFrame): The DataFrame to process.
    k (int): Number of clusters.
    scaler (optional): An instance of a Scaler from sklearn.preprocessing. Default is None.

    Returns:
    pd.DataFrame: Original DataFrame with an additional column 'Cluster' indicating the cluster assignment.

    Warnings:
        Throws UserWarning on Windows as KMeans is known to have a memory leak on Windows.
    """
    # Check if the DataFrame is valid
    if not isinstance(dataframe, pd.DataFrame):
        raise ValueError("Input is not a pandas DataFrame.")

    # Check if the DataFrame is empty
    if dataframe.empty:
        raise ValueError("Input DataFrame is empty.")

    # Validate k
    if not isinstance(k, int) or k <= 1:
        raise ValueError("k must be an integer greater than 1.")

    # Select numeric columns
    numeric_cols = dataframe.select_dtypes(include=['number']).columns
    if len(numeric_cols) == 0:
        raise ValueError("DataFrame contains no numeric columns.")

    numeric_data = dataframe[numeric_cols]

    # Apply scaling if a scaler is provided
    if scaler is not None:
        numeric_data = scaler.fit_transform(numeric_data)

    # Apply k-means clustering
    kmeans = KMeans(n_clusters=k, random_state=0, n_init=10)
    dataframe['Cluster'] = kmeans.fit_predict(numeric_data)

    return dataframe
