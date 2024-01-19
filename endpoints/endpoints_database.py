from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session

from db.db import (get_current_active_user, get_db, save_dataframe_to_database, get_dataframe_from_database,
                   delete_table_from_database, get_current_user)
from db.dbModels import UserInDB, UserTable, UploadedItemTable, User
from utils.convert_json_to_dataframe import convert_json_to_df
from utils.database_utils import extract_keys_of_dict, save_item_to_database

router = APIRouter()


# --------------------------------------Upload a table--------------------------------------#
# Endpoint to upload an item for the current user
@router.post("/users/me/item")
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


# ------------------------------------Get all tables------------------------------------#


# Endpoint to delete an item owned by the current user
@router.get("/users/me/items")
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
@router.delete("/users/me/item/{table_name}")
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


@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    return current_user
