import json

import requests

from api_functions.login_register import get_token
from utils.supervised_utils import validate_params, extract_model_and_encoder_format, upload_encoded_file, \
    upload_model_file, parser_supervised_response


def predict_with_supervised_algo(existing_model, table_name_to_train, algorithm, table_name_to_predict,
                                 name_of_function):
    """
    controller for the button "Supervised" in the gui

    :param existing_model: True or False (required)
    :param table_name_to_train: name of the table to train the model only needed if existing_model is False
    :param algorithm: name of the algorithm (required)
    :param table_name_to_predict: name of the table to predict the data(required)
    :param name_of_function: Test / Predict/ Save model button (required) > button > maybe redundant
    :return: the response of the server in json format
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
            # Upload model file, ending of model is depend on the algorithm > is  required
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
            # this is the table name of the training data, the table muss first
            # be uploaded over "Data import"
            "algorithm": algorithm,
            "table_name_to_predict": table_name_to_predict,
            # this is the table name of the data to predict, the table muss
            # first be uploaded over "Data import"
            "name_of_function": name_of_function  # Test / Predict/ Save model button > properly redundant
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
        print(e)
        return
