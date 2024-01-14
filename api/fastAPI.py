from datetime import timedelta
from io import StringIO

import numpy
import numpy as np
import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from starlette.responses import JSONResponse
from fastapi.encoders import jsonable_encoder


from MLModels.supervised import steel_classifier
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


# Common function to process data
async def process_supervised_data(dataframe_input: dict):
    data_input = dataframe_input.get("dataframe_input")
    df_json = data_input.get("data")

    df = await convert_json_to_df(df_json)

    # Call supervised model
    # TODO: mode train model or prediction!
    supervised_classifier = steel_classifier.SteelClassifier("../MLModels/supervised/test.h5",
                                                             "../MLModels/supervised/le.pkl",
                                                             "MLModels.supervised.deep_learning", "DeepLearning1", True)

    response_predict = []
    for index, row in df.iterrows():
        instance = row.values
        instance = np.char.mod('%s', instance)
        instance = np.char.replace(instance, ',', '.')
        instance = [float(i) for i in instance]
        instance = np.array(instance).reshape(1, -1)
        prediction = supervised_classifier.predict(instance)
        response_predict.append(prediction[0][0])

    # df_json = df.to_json(orient='split')

    response_data = {
        "prediction": response_predict
    }
    return response_data


async def convert_json_to_df(df_json):

    df_buffer = StringIO(df_json)
    df = pd.read_json(df_buffer, orient='split')
    return df


async def extract_model_options_and_df(data: dict):

    df_input = data.get("dataframe_input").get("data")
    df = await convert_json_to_df(df_input)

    model_options_input = data.get("model_options")

    return {
        "model_options": model_options_input,
        "data_frame": df
    }

# Endpoint for supervised prediction
@fast_app.post("/predict/supervised")
async def predict_supervised(dataframe_input: dict = None):
    # TODO: save the file in the database before making the prediction

    # file_name, file_path, file_size, df_data = await save_file(current_user, dataframe_input)
    # do = await save_item_to_database(current_user, db, file_name, file_path, file_size)
    # if do:
    #     # convert json to dataframe
    #     df_buffer = StringIO(df_data)
    #     df = pd.read_json(df_buffer, orient='split')
    #
    #     # Call supervised model
    return await process_supervised_data(dataframe_input)


# Endpoint for unsupervised prediction
#@fast_app.post("/predict_unsupervised")
#async def predict_unsupervised(dataframe_input: dict):
#    return await process_unsupervised_data(dataframe_input)

@fast_app.post("/predict/unsupervised/kmeans")
async def predict_unsupervised(data: dict):

    extracted_data = await extract_model_options_and_df(data)

    return cluster_Algorithm.kmeans(extracted_data["data_frame"], extracted_data["model_options"]["k_components"],
                                    extracted_data["model_options"]["scaler"]).to_json(orient="split")


@fast_app.post("/predict/unsupervised/gmm")
async def predict_unsupervised(data: dict):

    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.gmm_algorithm(extracted_data["data_frame"], extracted_data["model_options"]["k_components"])

    if isinstance(response_predict, pd.DataFrame):
        return response_predict.to_json(orient='split')
    else:
        return response_predict


@fast_app.post("/predict/unsupervised/hierarchical")
async def predict_unsupervised(data: dict):

    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.hierarchical(extracted_data["data_frame"], extracted_data["model_options"]["k_components"])
    return response_predict.to_json(orient='split')


@fast_app.post("/predict/unsupervised/tsne")
async def predict_unsupervised_tsne(data: dict):

    extracted_data = await extract_model_options_and_df(data)

    response_predict = cluster_Algorithm.tsne_algorithm(extracted_data["data_frame"], extracted_data["model_options"]["n_neighbours"], extracted_data["model_options"]["steps"], extracted_data["model_options"]["mode"])
    return response_predict


@fast_app.post("/predict/unsupervised/pca")
async def predict_unsupervised(data: dict):

    extracted_data = await extract_model_options_and_df(data)

    response_pca = cluster_Algorithm.pca(extracted_data["data_frame"], extracted_data["model_options"]["scaler"])
    return response_pca.tolist()


@fast_app.post("/predict/unsupervised/umap")
async def predict_unsupervised(data: dict):

    extracted_data = await extract_model_options_and_df(data)

    response_data = cluster_Algorithm.apply_umap(extracted_data["data_frame"], extracted_data["model_options"]["n_neighbours"], extracted_data["model_options"]["min_dist"], extracted_data["model_options"]["k_components"])
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
    file_name, namedb, file_size, df_data = await save_file(current_user, data_frame)
    # check if the namedb already exists
    item = db.query(UploadedItemTable).filter(UploadedItemTable.namedb == namedb).first()
    if item:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Item already exists",
            headers={"WWW-Authenticate": "Bearer"},
        )

    do = await save_item_to_database(current_user, db, file_name, namedb, file_size)
    if do:
        df = await convert_json_to_df(df_data)
        save_dataframe_to_database(df, namedb)
        return {"message": "Item uploaded successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
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


async def save_file(current_user_db, data_frame):
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
            status_code=status.HTTP_401_UNAUTHORIZED,
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


def extract_json_from_dic(dataframe_input):
    df_input = dataframe_input.get("dataframe_input")
    return df_input.get("data")
