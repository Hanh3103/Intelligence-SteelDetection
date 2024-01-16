from api_functions.unsupervised import predict_data_from_file_unsupervised_kmeans
from utils.convert_json_to_dataframe import convert_json_to_df
from utils.upload_csv_or_xlsx import return_dataframe_filesize_filename_from_csv_or_xlsx


async def extract_model_options_and_df(data: dict):
    df_input = data.get("dataframe_input").get("data")
    df = await convert_json_to_df(df_input)

    model_options_input = data.get("model_options")

    return {
        "model_options": model_options_input,
        "data_frame": df
    }


# Aktuell wird standardmäßig kmeans algorithmus aufgerufen bei unsupervised button click
# Kann man später lösen indem man hier ein switch case einbaut oder aus der gui gleich
# den jeweiligen Algorithmus aufruft.
# Testweise kann man hier auch die anderen unsupervised algorithmen aufrufen,
# diese sind ebenfalls weiter oben implementiert.
# Als Uploaddatei bitte test_data_unsupervised.csv aus dem Ordner /resources übergeben,
# da nur für diese Datei ein Output garantiert ist

def predict_data_from_file_unsupervised():
    # Function to predict data from an uploaded file in unsupervised mode
    # df, filesize, filename = upload_file()

    predict_data_from_file_unsupervised_kmeans(return_dataframe_filesize_filename_from_csv_or_xlsx(), 15, None)
