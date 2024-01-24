from api_functions.unsupervised import predict_data_from_file_unsupervised_kmeans
from api_functions.unsupervised import predict_data_from_file_unsupervised_affinity

from utils.convert_json_to_dataframe import convert_json_to_df
from utils.upload_csv_or_xlsx import return_dataframe_filesize_filename_from_csv_or_xlsx
from fastapi import HTTPException, status


async def extract_model_options_and_df(data: dict):
    """
    Extracts model options and DataFrame from input data.

    Parameters:
    - data: Dictionary containing input data.

    Returns:
    - Dictionary with keys "model_options" and "data_frame".
    """
    try:
        df_input = data.get("dataframe_input").get("data")
        df = await convert_json_to_df(df_input)

        model_options_input = data.get("model_options")

        return {
            "model_options": model_options_input,
            "data_frame": df
        }
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error extracting model options and DataFrame: {str(e)}")


# Currently, the kmeans algorithm is called by default when the unsupervised button is clicked.
# This can be solved later by implementing a switch case here or by directly calling the respective
# algorithm from the GUI.
# As a test, you can also call the other unsupervised algorithms here,
# which are also implemented above.
# For the upload file, please provide test_data_unsupervised.csv from the /resources folder,
# as only this file guarantees a specified output.

def predict_data_from_file_unsupervised():
    """
    Predicts data from an uploaded file in unsupervised mode using the kmeans algorithm.

    Returns:
    - None
    """
    try:
        # df, filesize, filename = upload_file()
        predict_data_from_file_unsupervised_affinity(return_dataframe_filesize_filename_from_csv_or_xlsx())
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error predicting data from file: {str(e)}")
