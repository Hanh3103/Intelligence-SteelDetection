import base64
import os
from tkinter import filedialog

import numpy as np
from fastapi import HTTPException
from starlette import status

from MLModels.supervised import steel_classifier
from utils.database_utils import get_dataframe_from_database
from enums.supervised_algo_enum import AlgorithmSupervised


def prediction_with_algorithm(classifier, table_to_predict_df, algorithm):
    """
    Make predictions using the specified algorithm.

    Args:
        classifier: The trained classifier model.
        table_to_predict_df: DataFrame containing data to predict.
        algorithm: The algorithm to use for prediction.

    Returns:
        response_predict: List of predicted values.
    """
    global response_predict
    if algorithm == AlgorithmSupervised.DeepLearning.value.get("algorithm"):
        # Preprocess data for deep learning algorithm
        response_predict = []
        for index, row in table_to_predict_df.iterrows():
            instance = row.values
            instance = np.char.mod('%s', instance)
            instance = np.char.replace(instance, ',', '.')
            instance = [float(i) for i in instance]
            instance = np.array(instance).reshape(1, -1)
            prediction = classifier.predict(instance)
            response_predict.append(prediction[0][0])
    elif algorithm == AlgorithmSupervised.Xgboost.value.get("algorithm"):
        # Predict using xgboost algorithm
        response_predict = classifier.predict(table_to_predict_df).tolist()

    return response_predict


# Endpoint for supervised prediction
async def extract_format_algorithm(algorithm):
    """
    Extract model and encoder format from the specified algorithm.

    Args:
        algorithm: The algorithm for which to extract the format.

    Returns:
        model_format: Format of the model file.
        encoder_format: Format of the encoder file.
    """
    for algo in AlgorithmSupervised:
        if algo.value.get("algorithm") == algorithm:
            model_format = algo.value.get("model_format")
            encoder_format = algo.value.get("encoder_format")
            return model_format, encoder_format


async def save_trained_model_co_on_server(algorithm, classifier, parent_dir):
    """
    Save the trained model and encoder on the server.

    Args:
        algorithm: The algorithm used for training.
        classifier: The trained classifier model.
        parent_dir: Parent directory for saving the model and encoder.

    Returns:
        name_of_model_to_save: Name of the saved model file.
        name_of_encoder_to_save: Name of the saved encoder file.
        encoded_string_model: Base64-encoded string of the saved model file.
        encoded_string_encoder: Base64-encoded string of the saved encoder file.
    """
    model_format, encoder_format = await extract_format_algorithm(algorithm)
    name_of_model_to_save = algorithm + model_format
    name_of_encoder_to_save = "encoder_" + algorithm + encoder_format
    path_to_save_model = os.path.join(parent_dir, "resources", "res_supervised", name_of_model_to_save)
    path_to_save_encoder = os.path.join(parent_dir, "resources", "res_supervised", name_of_encoder_to_save)

    # Save the model and encoder
    classifier.save_model(path_to_save_model, path_to_save_encoder)

    # Encode the file content as base64
    with open(path_to_save_model, "rb") as file:
        encoded_string_model = base64.b64encode(file.read()).decode("utf-8")
    with open(path_to_save_encoder, "rb") as file:
        encoded_string_encoder = base64.b64encode(file.read()).decode("utf-8")

    # Delete the model and encoder on the server
    await delete_model_and_encoder(path_to_save_model, path_to_save_encoder)

    return name_of_model_to_save, name_of_encoder_to_save, encoded_string_model, encoded_string_encoder


async def delete_model_and_encoder(path_of_model, path_of_encoder):
    """
    delete the model and encoder on server
    :param path_of_model: absolute path of the model on server
    :param path_of_encoder: absolute path of the encoder on server
    """
    if os.path.exists(path_of_model) and os.path.exists(path_of_encoder):
        os.remove(path_of_model)
        os.remove(path_of_encoder)


async def train_model_then_predict_classifier(class_name, encoded_file, module_name, parent_dir, table_name_to_train):
    """
    Train a classifier using the specified parameters and predict using the trained classifier.

    Args:
        class_name: The class name of the classifier.
        encoded_file: The uploaded file containing encoded data.
        module_name: The module name for the classifier.
        parent_dir: Parent directory for saving files.
        table_name_to_train: Name of the table to train the model.

    Returns:
        classifier: Trained classifier instance.
    """
    try:
        if encoded_file.filename == "":
            encoded_file_path = ""
        else:
            encoded_file_path = await save_uploaded_file(encoded_file, parent_dir)

        # Get the actual dataframe from the database
        table_to_train_df = get_dataframe_from_database(table_name_to_train)
        table_to_train_df.replace(',', '.', regex=True, inplace=True)

        # Load the classifier
        classifier = steel_classifier.SteelClassifier(table_to_train_df,
                                                      encoded_file_path,
                                                      module_name,
                                                      class_name,
                                                      False)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_417_EXPECTATION_FAILED,
            detail="Cannot set up the model " + str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return classifier


async def existing_model_classifier(class_name, encoded_file, module_name, parent_dir, upload_model_file):
    """
    Load an existing model classifier using the specified parameters.

    Args:
        class_name: The class name of the classifier.
        encoded_file: The uploaded file containing encoded data.
        module_name: The module name for the classifier.
        parent_dir: Parent directory for saving files.
        upload_model_file: The uploaded model file.

    Returns:
        classifier: Loaded classifier instance.
    """
    try:
        model_file_path = await save_uploaded_file(upload_model_file, parent_dir)

        # Encode file is optional
        if encoded_file.filename == "":
            encoded_file_path = ""
        else:
            encoded_file_path = await save_uploaded_file(encoded_file, parent_dir)

        # Load the classifier
        classifier = steel_classifier.SteelClassifier(model_file_path,
                                                      encoded_file_path,
                                                      module_name,
                                                      class_name,
                                                      True)

        # Delete the model and encoder on the server
        await delete_model_and_encoder(model_file_path, encoded_file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return classifier


async def save_uploaded_file(uploaded_file, parent_dir):
    """
    Save the uploaded file on the server.

    Args:
        uploaded_file: The file to be uploaded.
        parent_dir: Parent directory for saving files.

    Returns:
        uploaded_file_path: Path where the file is saved.
    """
    # Upload file
    uploaded_file_name = uploaded_file.filename
    uploaded_file_data = await uploaded_file.read()

    # Save file using the same name on the server
    uploaded_file_path = os.path.join(parent_dir, "resources", "res_supervised", uploaded_file_name)
    with open(uploaded_file_path, "wb") as buffer:
        buffer.write(uploaded_file_data)
    return uploaded_file_path


async def extract_modul_class_name(algorithm):
    """
    Extract module name and class name from the specified algorithm.

    Args:
        algorithm: The algorithm for which to extract names.

    Returns:
        module_name: The module name for the algorithm.
        class_name: The class name for the algorithm.
    """
    module_name = "MLModels.supervised." + algorithm
    class_name = ""
    for algo in AlgorithmSupervised:
        if algo.value.get("algorithm") == algorithm:
            class_name = algo.value.get("class_name")
    return module_name, class_name


async def parse_payload(dataframe_input):
    """
    Parse payload data to extract relevant information.

    Args:
        dataframe_input: The input dataframe containing relevant parameters.

    Returns:
        algorithm: The specified algorithm.
        existing_model: Flag indicating if an existing model is used.
        table_name_to_predict: The table name for prediction.
        table_name_to_train: The table name for training (if applicable).
        name_of_function: The name of the function.
    """
    existing_model = dataframe_input.get("existing_model")
    table_name_to_train = dataframe_input.get("table_name_to_train")
    algorithm = dataframe_input.get("algorithm")
    table_name_to_predict = dataframe_input.get("table_name_to_predict")
    name_of_function = dataframe_input.get("name_of_function")

    # Validate table_name_to_predict
    if table_name_to_predict == "":
        raise Exception("Please provide a table name to predict.")

    # Validate algorithm
    if algorithm == "":
        raise Exception("Please provide an algorithm.")

    return algorithm, existing_model, table_name_to_predict, table_name_to_train, name_of_function


def upload_file_with_file_ending(file_ending, text):
    """
    Upload a file with a specific file ending using a file dialog.

    Args:
        file_ending: The expected file ending (e.g., ".txt").
        text: The title of the file dialog.

    Returns:
        filename: The name of the selected file.
        file_data: The content of the selected file as bytes.
    """
    file_path = filedialog.askopenfilename(title=text)
    filename = file_path.split('/')[-1]

    # Validate file ending
    if file_path.endswith(file_ending):
        with open(file_path, 'rb') as file:
            file_data = file.read()

        return filename, file_data
    elif file_path == '':
        raise ValueError("No file selected.")
    else:
        raise Exception("Unsupported file type. Please provide" + file_ending + " file.")


def validate_params(algorithm, table_name_to_predict, existing_model, table_name_to_train):
    """
    Validate parameters for training or predicting.

    Args:
        algorithm: The specified algorithm.
        table_name_to_predict: The table name for prediction.
        existing_model: Flag indicating if an existing model is used.
        table_name_to_train: The table name for training (if applicable).

    Raises:
        Exception: If validation fails, an exception is raised with an appropriate message.
    """
    # Algorithm should not be empty and must be a valid algorithm
    if algorithm == "":
        raise Exception("Please provide an algorithm.")
    elif algorithm not in [algo.value.get("algorithm") for algo in AlgorithmSupervised]:
        raise Exception("Please provide a valid algorithm.")

    # table_name_to_predict should not be empty
    if table_name_to_predict == "":
        raise Exception("Please provide a table name to predict the data.")

    if existing_model == "False":
        # table_name_to_train should not be empty
        if table_name_to_train == "":
            raise Exception("Please provide a table name to train the model.")
    else:
        # table_name_to_train should be empty
        if table_name_to_train != "":
            raise Exception("Please leave the table name to train the model empty.")


def extract_model_and_encoder_format(algorithm):
    # Extract model and encoder format from algorithm
    for algo in AlgorithmSupervised:
        if algo.value.get("algorithm") == algorithm:
            model_format = algo.value.get("model_format")
            encoder_format = algo.value.get("encoder_format")
            return model_format, encoder_format


def upload_encoded_file(encoder_format, existing_model):
    """
    Upload an encoded file based on the specified encoder format.

    Args:
        encoder_format: The expected file format of the encoder file.
        existing_model: Flag indicating if an existing model is used.

    Returns:
        encoded_file_name: The name of the uploaded encoded file.
        encoded_file_data: The content of the uploaded encoded file as bytes.
    """
    try:
        # Upload encoded file
        text = "Upload encoder file"
        encoded_file_name, encoded_file_data = upload_file_with_file_ending(encoder_format, text)
    except ValueError as e:  # if no file is selected
        if existing_model == "True":  # the encoded file is required
            raise Exception(str(e) + " The encoded file is required.")
        else:  # the encoded file is optional
            encoded_file_name = ""
            encoded_file_data = b''
    return encoded_file_name, encoded_file_data


def upload_model_file(model_format):
    """
    Upload a model file based on the specified model format.

    Args:
        model_format: The expected file format of the model file.

    Returns:
        model_file_name: The name of the uploaded model file.
        model_file_data: The content of the uploaded model file as bytes.
    """
    # Upload model file
    text = "Upload model file"
    model_file_name, model_file_data = upload_file_with_file_ending(model_format, text)
    return model_file_name, model_file_data


def save_model_and_encoder(name_new_model, name_new_encoder, model_content, encoder_content):
    """
    Save the model and encoder files to a specified directory.

    Args:
        name_new_model: The name of the new model file.
        name_new_encoder: The name of the new encoder file.
        model_content: The content of the model file as base64-encoded string.
        encoder_content: The content of the encoder file as base64-encoded string.
    """
    file_path = filedialog.askdirectory(title="Select directory to save model and encoder")
    if file_path == '':
        print("No directory selected.")
        return
    else:
        # Save model
        try:
            with open(file_path + "/" + name_new_model, "wb") as file:
                file.write(base64.b64decode(model_content))
        except Exception as e:
            print(e)
            return

        print("Model saved under " + file_path + "/" + name_new_model)

        # Save encoder
        try:
            with open(file_path + "/" + name_new_encoder, "wb") as file:
                file.write(base64.b64decode(encoder_content))
        except Exception as e:
            print(e)
            return

        print("Encoder saved under " + file_path + "/" + name_new_encoder)


def parser_supervised_response(response, existing_model):
    """
    Parse the response from a supervised learning API and process the data.

    Args:
        response: The response object from the API.
        existing_model: Flag indicating if an existing model is used.
    """
    if response.status_code != 200:
        print(response.json().get("detail"))
    # Process the response data
    else:
        res = response.json()
        prediction = res.get("prediction")
        if existing_model == "False":
            test_result = res.get("test_results")
            print(test_result)
            name_new_model = res.get("name_new_model")
            name_new_encoder = res.get("name_new_encoder")
            model_content = res.get("encoded_model")
            assert model_content is not None
            encoder_content = res.get("encoded_encoder")

            # save model and encoder
            save_model_and_encoder(name_new_model, name_new_encoder, model_content, encoder_content)
        return prediction
