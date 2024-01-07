from io import StringIO
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from models import User, UserInDB, Token, TokenData, UserTable, UploadedItem, UploadedItemTable
import db as dataBase
from sqlalchemy.orm import Session
from api.readConfig import read_config_value
from typing_extensions import Annotated
from db import save_dataframe_to_database, get_dataframe_from_database

fast_app = FastAPI()


def get_db():
    db = dataBase.SessionLocal()
    try:
        yield db
    finally:
        db.close()


SECRET_KEY = read_config_value('TOKEN', 'SECRET_KEY')
ALGORITHM = read_config_value('TOKEN', 'ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(read_config_value('TOKEN', 'ACCESS_TOKEN_EXPIRE_MINUTES'))

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth_2_scheme = OAuth2PasswordBearer(tokenUrl="token")


@fast_app.get("/")
async def root():
    return {"message": "Hello World"}


@fast_app.post("/predict_supervised")
async def process_data(dataframe_input: dict):
    data_input = dataframe_input.get("dataframe_input")
    df_json = data_input.get("data")

    df = await convert_json_to_df(df_json)

    # save the dataframe into the database
    dataBase.save_dataframe_to_database(df, "OESdata")

    df_json = df.to_json(orient='split')

    response_data = {
        "input_value": dataframe_input,
        "statistics": df_json
    }
    return JSONResponse(content=response_data)

    # 1.) To be implemented: ML Algorithm on Dataframe

    # 2.) To be implemented: return result form Algorithm


async def convert_json_to_df(df_json):
    # Verwende StringIO, um die JSON-Zeichenfolge zu einem Datei-ähnlichen Puffer zu machen
    df_buffer = StringIO(df_json)
    # Jetzt können Sie df_buffer in ein DataFrame konvertieren
    df = pd.read_json(df_buffer, orient='split')
    return df


@fast_app.post("/predict_unsupervised")
async def process_data(dataframe_input: dict):
    data_input = dataframe_input.get("dataframe_input")
    df_json = data_input.get("data")

    # Verwende StringIO, um die JSON-Zeichenfolge zu einem Datei-ähnlichen Puffer zu machen
    df_buffer = StringIO(df_json)

    # Jetzt können Sie df_buffer in ein DataFrame konvertieren
    df = pd.read_json(df_buffer, orient='split')

    df_json = df.to_json(orient='split')

    response_data = {
        "input_value": dataframe_input,
        "statistics": df_json
    }
    return JSONResponse(content=response_data)

    # 1.) To be implemented: ML Algorithm on Dataframe

    # 2.) To be implemented: return result form Algorithm


def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


def get_password_hash(password):
    return pwd_context.hash(password)


def get_user(username: str, db: Session = Depends(get_db)):
    user = db.query(UserTable).filter(UserTable.username == username).first()
    if user:
        return UserInDB(
            username=user.username,
            full_name=user.fullname,
            email=user.email,
            password=user.password,
            disabled=user.disabled
        )
    return None


def authenticate_user(username: str, password: str, db: Session = Depends(get_db)):
    user = get_user(username, db)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


def create_access_token(data: dict, expires_delta: timedelta or None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta

    else:
        expire = datetime.utcnow + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, ALGORITHM)
    return encoded_jwt


async def get_current_user(token: Annotated[str, Depends(oauth_2_scheme)], db: Session = Depends(get_db)):
    credentialException = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                        detail="Could not validate credentials",
                                        headers={"WWW-Authenticate": "Bearer"})

    try:
        payload = jwt.decode(token, SECRET_KEY, ALGORITHM)
        username: str = payload.get("sub")
        if username is None:
            raise credentialException
        token_data = TokenData(username=username)
    except JWTError:
        raise credentialException

    user = get_user(token_data.username, db)
    if user is None:
        raise credentialException
    return user


async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


def check_if_user_name_unique(username: str, db: Session = Depends(get_db)):  # TODO: Douple code with get_user
    user = db.query(UserTable).filter(UserTable.username == username).first()
    if not user:
        return True
    else:
        return False


@fast_app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(form_data.username.lower(), form_data.password, db)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"}, )
    access_token_expires = timedelta(ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@fast_app.post("/registrate")
async def create_user(newUser: UserInDB, db: Session = Depends(get_db)):
    isUnique = check_if_user_name_unique(newUser.username, db)

    if isUnique:
        new_user = UserTable(
            username=newUser.username.lower(),
            fullname=newUser.full_name,
            email=newUser.email,
            password=get_password_hash(newUser.password),
            disabled=newUser.disabled
        )

        db.add(new_user)
        db.commit()
        return {"message": "User created successfully"}
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Username already exists",
                            headers={"WWW-Authenticate": "Bearer"}, )


@fast_app.get("/users/me", response_model=User)
async def read_users_me(current_user: UserInDB = Depends(get_current_active_user)):
    return current_user


# Endpoint to upload an item for the current user
@fast_app.post("/users/me/item")
async def upload_own_item(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        data_frame: dict = None
):
    file_name, namedb, file_size, df_data = await save_file(current_user, data_frame)
    print(file_name, namedb, file_size, df_data)
    do = await save_item_to_database(current_user, db, file_name, namedb, file_size)
    if do:
        df = await convert_json_to_df(df_data)
        save_dataframe_to_database(df, namedb)
        return {"message": "Item uploaded successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Item could not be uploaded",
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

    # get all tables from the database with the same name as the items
    list_tables = []
    for item in items:
        table = get_dataframe_from_database(item.namedb)
        list_tables.append(table)

    # convert the tables to json
    list_json = []
    for table in list_tables:
        table_json = table.to_json(orient='split')
        list_json.append(table_json)

    # create a json response
    response = JSONResponse(content=list_json)
    return response

# @fast_app.get("/users/me/items")
# async def read_own_items(db: Session = Depends(get_db),
#                          current_user: UserInDB = Depends(get_current_active_user),
#                          skip: int = 0,
#                          limit: int = 10) -> list:
#     current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()
#     items = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user_db.id).offset(skip).limit(limit).all()
#
#     #create a list of dictionaries from the items
#     items = [item.__dict__ for item in items]
#     items = [{key: value for key, value in item.items() if key != "_sa_instance_state"} for item in items]
#
#     # get the file from item.filepath and add it to the dictionary
#     for item in items:
#         with open(item["filepath"], "r") as file:
#             item["data"] = file.read()
#
#     return items
#
#
# @fast_app.post("/users/me/item")
# async def upload_own_item(db: Session = Depends(get_db),
#                           current_user: UserInDB = Depends(get_current_active_user),
#                           data_frame: dict = None):
#     current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()
#     data_input = data_frame.get("dataframe_input")
#     df_json = data_input.get("data")
#     file_name = data_input.get("filename")
#     file_size = data_input.get("filesize")
#
#     # save the df_json locally, each user has its own folder, then save the path of the file to the database
#     import os
#     import json
#     from api.readConfig import read_config_value
#     user_directory = read_config_value("UPLOAD", "UPLOAD_FOLDER") + "/" + current_user_db.username
#     os.makedirs(user_directory, exist_ok=True)
#
#     file_path = user_directory + "/" + file_name
#     with open(file_path, "w") as file:
#         json.dump(df_json, file)
#
#     new_item = UploadedItemTable(
#         owner_id=current_user_db.id,
#         filename=file_name,
#         filesize=file_size,
#         filepath=file_path
#     )
#
#     db.add(new_item)
#     db.commit()
#     return {"message": "Item uploaded successfully"}
