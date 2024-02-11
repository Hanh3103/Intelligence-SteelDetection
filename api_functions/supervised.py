import json

import requests

from api_functions.login_register import get_token
from utils.supervised_utils import validate_params, extract_model_and_encoder_format, upload_encoded_file, \
    upload_model_file, parser_supervised_response


def predict_with_supervised_algo(existing_model, table_name_to_train, algorithm, table_name_to_predict,
                                 name_of_function):
    """
    Controller for the button "Supervised" in the GUI.

    Args:
        existing_model (bool): True or False (required).
        table_name_to_train (str): Name of the table to train the model, only needed if existing_model is False.
        algorithm (str): Name of the algorithm (required).
        table_name_to_predict (str): Name of the table to predict the data (required).
        name_of_function (str): Test / Predict/ Save model button (required) > button > may be redundant.

    Returns:
        dict: The response of the server in JSON format. Returns None in case of an exception.

    Input:
        - The function uses the global function `get_token()` to retrieve the access token.
        - Calls `validate_params`, `extract_model_and_encoder_format`, `upload_encoded_file`, `upload_model_file`,
          and `parser_supervised_response` as part of its execution.

    Output:
        dict: The response of the server in JSON format. Returns None in case of an exception.
    """
    token_save = get_token()
    # Validate params
    try:
        validate_params(algorithm, table_name_to_predict, existing_model, table_name_to_train)
        model_format, encoder_format = extract_model_and_encoder_format(algorithm)

        # Upload files
        if existing_model == "False":  # train new model and predict
            # upload encoded file > is optional
            encoded_file_name, encoded_file_data = upload_encoded_file(encoder_format, existing_model)

            files = {"upload_model_file": ("", b''),
                     "encoded_file": (encoded_file_name, encoded_file_data)}

        else:  # use upload existing model and predict
            # Upload model file, ending of model is depend on the algorithm > is required
            model_file_name, model_file_data = upload_model_file(model_format)
            # Upload encoded file > is optional (actually not optional)
            encoded_file_name, encoded_file_data = upload_encoded_file(encoder_format, existing_model)

            # Include the uploaded files in the request payload
            files = {"upload_model_file": (model_file_name, model_file_data),
                     "encoded_file": (encoded_file_name, encoded_file_data)}

        # Include parameters in the request payload
        payload = {
            "existing_model": existing_model,
            "table_name_to_train": table_name_to_train,
            "algorithm": algorithm,
            "table_name_to_predict": table_name_to_predict,
            "name_of_function": name_of_function  # Test / Predict/ Save model button > may be redundant
        }
        payload = json.dumps(payload)

        data = {
            "up_model_options": payload
        }
        header = {"accept": "application/json", "Authorization": f"Bearer {token_save}"}
        response = requests.post("http://127.0.0.1:8000/predict/supervised", files=files,
                                 data=data, headers=header)
        return parser_supervised_response(response, existing_model)
    except Exception as e:
        # Print the exception and return None
        print(e)
        return None
