import base64
import os
from tkinter import filedialog
import logging

import numpy as np
from fastapi import HTTPException
from starlette import status

from MLModels.supervised import steel_classifier
from db.db import get_dataframe_from_database
from enums.supervised_algo_enum import AlgorithmSupervised


def prediction_with_algorithm(classifier, table_to_predict_df, algorithm):
    global response_predict
    if algorithm == AlgorithmSupervised.DeepLearning.value.get("algorithm"):
        # this algorithm takes only numpy array as input
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
        # xgboost takes only dataframe as input
        response_predict = classifier.predict(table_to_predict_df).tolist()

    return response_predict


# Endpoint for supervised prediction


async def extract_format_algorithm(algorithm):
    # Extract model and encoder format from algorithm
    for algo in AlgorithmSupervised:
        if algo.value.get("algorithm") == algorithm:
            model_format = algo.value.get("model_format")
            encoder_format = algo.value.get("encoder_format")
            return model_format, encoder_format


async def save_trained_model_co_on_server(algorithm, classifier, parent_dir):
    model_format, encoder_format = await extract_format_algorithm(algorithm)
    name_of_model_to_save = algorithm + model_format
    name_of_encoder_to_save = "encoder_" + algorithm + encoder_format
    path_to_save_model = os.path.join(parent_dir, "resources", "res_supervised", name_of_model_to_save)
    path_to_save_encoder = os.path.join(parent_dir, "resources", "res_supervised", name_of_encoder_to_save)
    classifier.save_model(path_to_save_model, path_to_save_encoder)
    # Encode the file content as base64
    with open(path_to_save_model, "rb") as file:
        encoded_string_model = base64.b64encode(file.read()).decode("utf-8")
    with open(path_to_save_encoder, "rb") as file:
        encoded_string_encoder = base64.b64encode(file.read()).decode("utf-8")

    # delete the model and encoder on server
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
    try:
        if encoded_file.filename == "":
            encoded_file_path = ""
        else:
            encoded_file_path = await save_uploaded_file(encoded_file, parent_dir)

        # get the actual dataframe from the database
        table_to_train_df = get_dataframe_from_database(table_name_to_train)
        table_to_train_df.replace(',', '.', regex=True, inplace=True)
        # load the classifier
        classifier = steel_classifier.SteelClassifier(table_to_train_df,
                                                      encoded_file_path,
                                                      module_name,
                                                      class_name,
                                                      False)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_417_EXPECTATION_FAILED,
            detail="Cannot setup the model " + str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return classifier


async def existing_model_classifier(class_name, encoded_file, module_name, parent_dir, upload_model_file):
    try:
        model_file_path = await save_uploaded_file(upload_model_file, parent_dir)
        # encode file is optional !!! > it can be empty "except for deep learning in case exist_model is true"
        if encoded_file.filename == "":
            encoded_file_path = ""
        else:
            encoded_file_path = await save_uploaded_file(encoded_file, parent_dir)

        # load the classifier
        classifier = steel_classifier.SteelClassifier(model_file_path,
                                                      encoded_file_path,
                                                      module_name,
                                                      class_name,
                                                      True)
        await delete_model_and_encoder(model_file_path, encoded_file_path)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return classifier  #


async def save_uploaded_file(uploaded_file, parent_dir):
    # Upload file
    uploaded_file_name = uploaded_file.filename
    uploaded_file_data = await uploaded_file.read()
    # Save file using the same name on the server
    uploaded_file_path = os.path.join(parent_dir, "resources", "res_supervised", uploaded_file_name)
    with open(uploaded_file_path, "wb") as buffer:
        buffer.write(uploaded_file_data)
    return uploaded_file_path


async def extract_modul_class_name(algorithm):
    # Get the module name and class name from the algorithm
    module_name = "MLModels.supervised." + algorithm
    class_name = ""
    for algo in AlgorithmSupervised:
        if algo.value.get("algorithm") == algorithm:
            class_name = algo.value.get("class_name")
    return module_name, class_name


async def parse_payload(dataframe_input):
    existing_model = dataframe_input.get("existing_model")
    table_name_to_train = dataframe_input.get("table_name_to_train")
    algorithm = dataframe_input.get("algorithm")
    table_name_to_predict = dataframe_input.get("table_name_to_predict")
    name_of_function = dataframe_input.get("name_of_function")
    if table_name_to_predict == "":
        raise Exception("Please provide a table name to predict.")
    if algorithm == "":
        raise Exception("Please provide an algorithm.")
    return algorithm, existing_model, table_name_to_predict, table_name_to_train, name_of_function


def upload_file_with_file_ending(file_ending, text):
    file_path = filedialog.askopenfilename(title=text)

    filename = file_path.split('/')[-1]

    if file_path.endswith(file_ending):
        with open(file_path, 'rb') as file:
            file_data = file.read()

        return filename, file_data
    elif file_path == '':
        raise ValueError("No file selected.")
    else:
        raise Exception("Unsupported file type. Please provide" + file_ending + " file.")


def validate_params(algorithm, table_name_to_predict, existing_model, table_name_to_train):
    # Algorithm and table_name_to_predict should not be empty independently of mode_train
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
    # Upload model file
    text = "Upload model file"
    model_file_name, model_file_data = upload_file_with_file_ending(model_format, text)
    return model_file_name, model_file_data


def save_model_and_encoder(name_new_model, name_new_encoder, model_content, encoder_content):
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
    if response.status_code != 200:
        print(response.json().get("detail"))
    # Process the response data
    else:
        res = response.json()
        prediction = res.get("prediction")
        print(prediction)
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
