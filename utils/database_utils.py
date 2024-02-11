# Import necessary modules
from datetime import datetime, timedelta
from typing import Optional

import pandas as pd
from fastapi import Depends, HTTPException
from jose import jwt, JWTError
from pandasgui import show
from sqlalchemy import MetaData, Table
from sqlalchemy.orm import Session
from starlette import status

from db.dbModels import UploadedItemTable
from db.dbModels import UserInDB, UserTable, TokenData
from db.db_instance import get_current_active_engine
from db.db_instance import get_current_active_db_session as get_db
from db.db_instance import oauth2_scheme as oauth2_scheme
from utils.constants import SECRET_KEY, ALGORITHM


# Asynchronous function to save an item to the database
async def save_item_to_database(current_user, db, file_name, name_db, file_size):
    """
    Saves an item to the database.

    Parameters:
    - current_user: Current user object.
    - db: Database session.
    - file_name: Name of the file.
    - name_db: Database name.
    - file_size: Size of the file.

    Returns:
    - True if the item is successfully saved to the database, otherwise False.
    """
    try:
        # Retrieve the user's database entry
        current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()

        # Create a new item entry in the UploadedItemTable
        new_item = UploadedItemTable(
            owner_id=current_user_db.id,
            filename=file_name,
            filesize=file_size,
            namedb=name_db
        )

        # Add the new item to the database and commit the changes
        db.add(new_item)
        db.commit()

        # Return True to indicate successful item addition
        return True
    except Exception as e:
        # Print the exception, rollback changes, and return False in case of an error
        print(e)
        db.rollback()
        return False


# Asynchronous function to extract keys from a dictionary
async def extract_keys_of_dict(current_user_db, data_frame):
    """
    Extracts keys from a dictionary containing data frame information.

    Parameters:
    - current_user_db: User database entry.
    - data_frame: Dictionary containing data frame information.

    Returns:
    - Tuple containing extracted values: (file_name, namedb, file_size, df_json).
    """
    data_input = data_frame.get("data_frame")
    df_json = data_input.get("data")
    file_name = data_input.get("filename")
    file_size = data_input.get("filesize")

    namedb = current_user_db.username + "_" + file_name
    return file_name, namedb, file_size, df_json


# Function to show dataframes using the pandasgui library (not elegant!)
def show_dataframes(dict_dataframes):
    """
    Displays dataframes using the pandasgui library.

    Parameters:
    - dict_dataframes: Dictionary of dataframes.
    """
    show(**dict_dataframes)


# Function to save a DataFrame to a PostgresSQL table
def save_dataframe_to_database(df_in, table_name_db):

    """
    Save a DataFrame to a PostgresSQL table.

    Args:
        df_in (pd.DataFrame): DataFrame to be saved.
        table_name_db (str): Name of the PostgresSQL table.

    Returns:
        None.
    """
    engine = get_current_active_engine()
    df_in.to_sql(table_name_db, engine, if_exists="replace", index=False)


# Function to retrieve a DataFrame from a database table
def get_dataframe_from_database(table_name_db):
    """
    Retrieve a DataFrame from a PostgresSQL table.

    Args:
        table_name_db (str): Name of the PostgresSQL table.

    Returns:
        pd.DataFrame: Retrieved DataFrame.
    """
    engine = get_current_active_engine()
    return pd.read_sql_table(table_name_db, engine)


# Function to delete a table from the database
def delete_table_from_database(table_name_db):
    """
    Delete a table from the PostgresSQL database.

    Args:
        table_name_db (str): Name of the PostgresSQL table.

    Returns:
        None.
    """
    metadata = MetaData()
    engine = get_current_active_engine()
    table_to_delete = Table(table_name_db, metadata, autoload_with=engine, extend_existing=True)
    table_to_delete.drop(engine, checkfirst=True)


# Function to create an access token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    """
    Create an access token.

    Args:
        data (dict): Data to be encoded in the token.
        expires_delta (Optional[timedelta]): Optional expiration time.

    Returns:
        str: Created access token.
    """
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# Function to get the current user based on the JWT token
async def get_current_user(token: str = Depends(oauth2_scheme), db_session: Session = Depends(get_db)):
    """
    Get the current user based on the JWT token.

    Args:
        token (str): JWT token.
        db_session (Session): Database session.

    Returns:
        UserTable: Current user information.

    Raises:
        HTTPException: If credentials cannot be validated.
    """
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    user_dict = get_user(token_data.username, db_session)
    if user_dict is None:
        raise credentials_exception

    return user_dict


# Function to get the current active user
async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    """
    Get the current active user.

    Args:
        current_user (UserInDB): Current user information.

    Raises:
        HTTPException: If the user is inactive.
    """
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Function to get a user from the database based on the username
def get_user(username: str, db_session: Session = Depends(get_db)):
    """
    Get a user from the database based on the username.

    Args:
        username (str): Username to search for.
        db_session (Session): Database session.

    Returns:
        Tuple[UserTable, str] or None: User information and hashed password if found, None otherwise.
    """

    return db_session.query(UserTable).filter(UserTable.username == username).first()
