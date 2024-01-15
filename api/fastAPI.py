import os
from datetime import timedelta
from io import StringIO

import numpy
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status, File, UploadFile, Form
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse
from fastapi.encoders import jsonable_encoder
from pydantic import Json

from MLModels.supervised import steel_classifier
from MLModels.supervised.supervised_algo_enum import AlgorithmSupervised
from db.db import (authenticate_user, create_access_token, check_if_user_name_unique, get_password_hash, \
                   get_current_active_user, get_db, save_dataframe_to_database, get_dataframe_from_database,
                   get_current_user,
                   delete_table_from_database)
from MLModels.unsupervised import cluster_Algorithm
from enums import unsupervised_models_enum
from db.db import authenticate_user, create_access_token, check_if_user_name_unique, get_password_hash, \
    get_current_active_user, get_db, save_dataframe_to_database, get_dataframe_from_database, get_current_user
from db.dbModels import User, UserInDB, Token, UserTable, UploadedItemTable
from utils import readConfig  # Importing a utility function for reading configuration

# Create a FastAPI app instance
fast_app = FastAPI()

# Read configuration values
SECRET_KEY = readConfig.read_config_value('TOKEN', 'SECRET_KEY')
ALGORITHM = readConfig.read_config_value('TOKEN', 'ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(readConfig.read_config_value('TOKEN', 'ACCESS_TOKEN_EXPIRE_MINUTES'))


# Root endpoint
@fast_app.get("/")
async def root():
    return {"message": "Hello World"}

# ------------------------------------SuperVised------------------------------------#
# Endpoint for supervised prediction
def prediction_result(classifier, table_to_predict_df, algorithm):
    global response_predict
    if algorithm == AlgorithmSupervised.DeepLearning.value[0]:
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
    elif algorithm == AlgorithmSupervised.Xgboost.value[0]:
        # xgboost takes only dataframe as input
        response_predict = classifier.predict(table_to_predict_df)

    return response_predict


async def convert_json_to_df(df_json):
    df_buffer = StringIO(df_json)
    df = pd.read_json(df_buffer, orient='split')
    return df

# Endpoint for supervised prediction
@fast_app.post("/predict/supervised")
async def predict_supervised(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        upload_model_file: UploadFile = File(...),
        encoded_file: UploadFile = File(...),
        up_model_options: Json = Form(...)
):
    # parse the payload
    try:
        algorithm, existing_model, table_name_to_predict, table_name_to_train, name_of_function = await parse_payload(
            up_model_options)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="parse payload problem" + str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    # get the actual dataframe from the database
    try:
        table_to_predict_df = get_dataframe_from_database(table_name_to_predict)
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Table to predict not found in database " + str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )

    parent_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

    module_name, class_name = await extract_modul_class_name(algorithm)

    if existing_model == "True":
        classifier = await existing_model_classifier(class_name, encoded_file, module_name, parent_dir,
                                                     upload_model_file)
        # only prediction works !!!
        try:
            results = prediction_result(classifier, table_to_predict_df, algorithm)
            return {"prediction": results}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="prediction existing model has problems" + str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )
    else:
        classifier = await train_model_then_predict(class_name, encoded_file, module_name, parent_dir,
                                                    table_name_to_train)
        # prediction then save the model then test the model
        try:
            results = prediction_result(classifier, table_to_predict_df, algorithm)
            print(type(results))
            print(results)
            test_results = classifier.test_model()
            print(test_results)
            print(type(test_results))
            name_of_model_to_save = algorithm + [algo.value[1] for algo in AlgorithmSupervised
                                                 if algo.value[0] == algorithm][0]
            name_of_encoder_to_save = "encoder_" + algorithm + ".pkl"
            classifier.save_model(name_of_model_to_save, name_of_encoder_to_save)
            return {"prediction": results, "test_results": test_results}
        except Exception as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="prediction and saving have problems " + str(e),
                headers={"WWW-Authenticate": "Bearer"},
            )


async def train_model_then_predict(class_name, encoded_file, module_name, parent_dir, table_name_to_train):
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
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail=str(e),
            headers={"WWW-Authenticate": "Bearer"},
        )
    return classifier


async def save_uploaded_file(uploaded_file, parent_dir):
    # Upload file
    uploaded_file_name = uploaded_file.filename
    uploaded_file_data = await uploaded_file.read()
    # Save file using the same name on the server
    uploaded_file_path = os.path.join(parent_dir, "MLModels", "supervised", uploaded_file_name)
    with open(uploaded_file_path, "wb") as buffer:
        buffer.write(uploaded_file_data)
    return uploaded_file_path


async def extract_modul_class_name(algorithm):
    # Get the module name and class name from the algorithm
    module_name = "MLModels.supervised." + algorithm
    class_name = ""
    if algorithm == AlgorithmSupervised.DeepLearning.value[0]:
        class_name = AlgorithmSupervised.DeepLearning.value[2]
    elif algorithm == AlgorithmSupervised.Xgboost.value[0]:
        class_name = AlgorithmSupervised.Xgboost.value[2]
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


# ------------------------------------UnSuperVised------------------------------------#
async def extract_model_options_and_df(data: dict):
    df_input = data.get("dataframe_input").get("data")
    df = await convert_json_to_df(df_input)

    model_options_input = data.get("model_options")

    return {
        "model_options": model_options_input,
        "data_frame": df
    }


@fast_app.post("/predict/unsupervised/kmeans")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    return cluster_Algorithm.kmeans(extracted_data["data_frame"], extracted_data["model_options"]["k_components"],
                                    extracted_data["model_options"]["scaler"]).to_json(orient="split")


@fast_app.post("/predict/unsupervised/gmm")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.gmm_algorithm(extracted_data["data_frame"],
                                                       extracted_data["model_options"]["k_components"])

    if isinstance(response_predict, pd.DataFrame):
        return response_predict.to_json(orient='split')
    else:
        return response_predict


@fast_app.post("/predict/unsupervised/hierarchical")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.hierarchical(extracted_data["data_frame"],
                                                      extracted_data["model_options"]["k_components"])
    return response_predict.to_json(orient='split')


@fast_app.post("/predict/unsupervised/tsne")
async def predict_unsupervised_tsne(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.tsne_algorithm(extracted_data["data_frame"],
                                                        extracted_data["model_options"]["n_neighbours"],
                                                        extracted_data["model_options"]["steps"],
                                                        extracted_data["model_options"]["mode"])
    return response_predict


@fast_app.post("/predict/unsupervised/pca")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_pca = cluster_Algorithm.pca(extracted_data["data_frame"], extracted_data["model_options"]["scaler"])
    return response_pca.tolist()


@fast_app.post("/predict/unsupervised/umap")
async def predict_unsupervised(data: dict):
    extracted_data = await extract_model_options_and_df(data)

    response_data = cluster_Algorithm.apply_umap(extracted_data["data_frame"],
                                                 extracted_data["model_options"]["n_neighbours"],
                                                 extracted_data["model_options"]["min_dist"],
                                                 extracted_data["model_options"]["k_components"])
    return response_data.tolist()


# Token endpoint for user authentication
@fast_app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(form_data.username.lower(), form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@fast_app.post("/registrate")
async def create_user(new_user: UserInDB, db: Session = Depends(get_db)):
    is_unique = check_if_user_name_unique(new_user.username, db)

    if is_unique:
        new_user = UserTable(
            username=new_user.username.lower(),
            fullname=new_user.full_name,
            email=new_user.email,
            password=get_password_hash(new_user.password),
            disabled=new_user.disabled
        )

        db.add(new_user)
        db.commit()
        return {"message": "User created successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username already exists",
            headers={"WWW-Authenticate": "Bearer"},
        )


# Endpoint to get current user
@fast_app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user


# Endpoint to upload an item for the current user
@fast_app.post("/users/me/item")
async def upload_own_item(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        data_frame: dict = None
):
    file_name, namedb, file_size, df_data = await extract_keys_of_dict(current_user, data_frame)

    # check if the number of items in database less than 20
    items = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user.id).all()
    if len(items) >= 20:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Maximum number of items reached, please delete some items",
            headers={"WWW-Authenticate": "Bearer"},
        )
    else:
        # if the number of items in database less than 20, then check if the file name already exists
        # check if the filename already exists
        items = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user.id).filter(
            UploadedItemTable.filename == file_name).all()

        # check if the file size is the same as the one in the database, which means that the file already exists
        for item in items:
            if item.filesize == file_size:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="File already exists",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # if the file name already exists, but there is a different file size, then add the number of items with the
        # same name
        if len(items) > 0:
            namedb = namedb + "_" + str(len(items))
            file_name = file_name + "_" + str(len(items))

        do = await save_item_to_database(current_user, db, file_name, namedb, file_size)
        if do:
            df = await convert_json_to_df(df_data)
            save_dataframe_to_database(df, namedb)
            return {"message": "Item uploaded successfully"}
        else:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Item could not be uploaded successfully due to an error in the database",
                headers={"WWW-Authenticate": "Bearer"},
            )


async def save_item_to_database(current_user, db, file_name, name_db, file_size):
    try:
        current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()
        new_item = UploadedItemTable(
            owner_id=current_user_db.id,
            filename=file_name,
            filesize=file_size,
            namedb=name_db
        )
        db.add(new_item)
        db.commit()
        return True
    except Exception as e:
        print(e)
        db.rollback()
        return False


async def extract_keys_of_dict(current_user_db, data_frame):
    data_input = data_frame.get("data_frame")
    df_json = data_input.get("data")
    file_name = data_input.get("filename")
    file_size = data_input.get("filesize")

    namedb = current_user_db.username + "_" + file_name
    return file_name, namedb, file_size, df_json


# Endpoint to delete an item owned by the current user
@fast_app.get("/users/me/items")
async def read_own_items(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        skip: int = 0,
        limit: int = 10
) -> JSONResponse:
    current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()
    items = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user_db.id).offset(skip).limit(
        limit).all()

    if not items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No items found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # get all tables from the database with the same name as the items
    dict_tables = {}
    for item in items:
        table = get_dataframe_from_database(item.namedb)
        dict_tables[item.namedb] = table

    # convert the tables to json
    list_json = []
    for name, table in dict_tables.items():
        table_dict_one = {}
        table_json = table.to_json(orient='split')
        table_dict_one["name"] = name
        table_dict_one["data"] = table_json
        list_json.append(table_dict_one)

    # create a json response
    response = JSONResponse(content=list_json)
    return response


# ------------------------------------Delete a table with table name------------------------------------#
# Endpoint to delete an item owned by the current user
@fast_app.delete("/users/me/item/{table_name}")
async def delete_own_item(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        table_name: str = None
):
    current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()
    item = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user_db.id).filter(
        UploadedItemTable.namedb == table_name).first()
    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    else:
        delete_table_from_database(table_name)
        try:
            db.delete(item)
            db.commit()
            return {"message": "Item deleted successfully"}
        except Exception as e:
            print(e)
            db.rollback()
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Item could not be deleted successfully due to an error in the database",
                headers={"WWW-Authenticate": "Bearer"},
            )
