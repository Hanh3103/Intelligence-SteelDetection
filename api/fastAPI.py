from datetime import timedelta
from io import StringIO
import os
import json

import pandas as pd
from fastapi import FastAPI, Depends, HTTPException, status
from fastapi.responses import JSONResponse
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from db.db import authenticate_user, create_access_token, check_if_user_name_unique, get_password_hash, \
    get_current_active_user, get_db
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
async def process_data(dataframe_input: dict):
    data_input = dataframe_input.get("dataframe_input")
    df_json = data_input.get("data")

    df_buffer = StringIO(df_json)
    df = pd.read_json(df_buffer, orient='split')

    df_json = df.to_json(orient='split')

    response_data = {
        "input_value": dataframe_input,
        "statistics": df_json
    }
    return JSONResponse(content=response_data)


# Endpoint for supervised prediction
@fast_app.post("/predict_supervised")
async def predict_supervised(dataframe_input: dict):
    return await process_data(dataframe_input)


# Endpoint for unsupervised prediction
@fast_app.post("/predict_unsupervised")
async def predict_unsupervised(dataframe_input: dict):
    return await process_data(dataframe_input)


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


# Registration endpoint for creating a new user
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


# Endpoint to get information about the current user
@fast_app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user


# Endpoint to get items owned by the current user
@fast_app.get("/users/me/items")
async def read_own_items(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        skip: int = 0,
        limit: int = 10
) -> list:
    current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()
    items = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user_db.id).offset(skip).limit(
        limit).all()

    items = [
        {key: value for key, value in item.__dict__.items() if key != "_sa_instance_state"}
        for item in items
    ]

    for item in items:
        with open(item["filepath"], "r") as file:
            item["data"] = file.read()

    return items


# Endpoint to upload an item for the current user
@fast_app.post("/users/me/item")
async def upload_own_item(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        data_frame: dict = None
):
    current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()
    data_input = data_frame.get("dataframe_input")
    df_json = data_input.get("data")
    file_name = data_input.get("filename")
    file_size = data_input.get("filesize")

    user_directory = read_config_value("UPLOAD", "UPLOAD_FOLDER") + "/" + current_user_db.username
    os.makedirs(user_directory, exist_ok=True)

    file_path = user_directory + "/" + file_name
    with open(file_path, "w") as file:
        json.dump(df_json, file)

    new_item = UploadedItemTable(
        owner_id=current_user_db.id,
        filename=file_name,
        filesize=file_size,
        filepath=file_path
    )

    db.add(new_item)
    db.commit()
    return {"message": "Item uploaded successfully"}
