from fastapi import Depends, HTTPException, status, APIRouter
from fastapi.responses import JSONResponse
from sqlalchemy.orm import Session
from utils.database_utils import (get_current_active_user, save_dataframe_to_database, get_dataframe_from_database,
                            delete_table_from_database, get_current_user)
from db.db_instance import get_current_active_db_session as get_db
from db.dbModels import UserInDB, UserTable, UploadedItemTable, User
from utils.convert_json_to_dataframe import convert_json_to_df
from utils.database_utils import extract_keys_of_dict, save_item_to_database

router = APIRouter()


# Endpoint to upload an item for the current user
@router.post("/users/me/item")
async def upload_own_item(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        data_frame: dict = None
):
    """
    Endpoint to upload an item for the current authenticated user.

    Args:
        db (Session): Database session dependency.
        current_user (UserInDB): Current authenticated user dependency.
        data_frame (dict): Data containing file information and DataFrame.

    Returns:
        dict: Response indicating success or failure.

    Raises:
        HTTPException: Raised for various HTTP status codes with corresponding details.

    Notes:
        - Checks if the user has reached the maximum number of allowed items.
        - Checks if the file with the same name and size already exists.
        - Handles scenarios where the file name already exists but with a different size.
        - Saves the item information and DataFrame to the database.
    """
    # Extract file information and DataFrame
    file_name, namedb, file_size, df_data = await extract_keys_of_dict(current_user, data_frame)

    # Check if the user has reached the maximum number of items
    items = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user.id).all()
    if len(items) >= 20:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Maximum number of items reached, please delete some items",
            headers={"WWW-Authenticate": "Bearer"},
        )
    else:
        # Check if the file name already exists
        items = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user.id).filter(
            UploadedItemTable.filename == file_name).all()

        # Check if the file size is the same as the one in the database, indicating that the file already exists
        for item in items:
            if item.filesize == file_size:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail="File already exists",
                    headers={"WWW-Authenticate": "Bearer"},
                )

        # If the file name already exists but with a different file size, then append a suffix
        if len(items) > 0:
            namedb = namedb + "_" + str(len(items))
            file_name = file_name + "_" + str(len(items))

        # Save the item information and DataFrame to the database
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


# Endpoint to get all tables
@router.get("/users/me/items")
async def read_own_items(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        skip: int = 0,
        limit: int = 10
) -> JSONResponse:
    """
    Endpoint to retrieve items owned by the current authenticated user.

    Args:
        db (Session): Database session dependency.
        current_user (UserInDB): Current authenticated user dependency.
        skip (int): Number of items to skip in the query.
        limit (int): Maximum number of items to retrieve.

    Returns:
        JSONResponse: Response containing information about the retrieved tables.

    Raises:
        HTTPException: Raised for various HTTP status codes with corresponding details.

    Notes:
        - Retrieves items owned by the user from the database.
        - Retrieves tables associated with the items from the database.
        - Converts tables to JSON and constructs a response.
    """
    # Get the current user from the database
    current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()

    # Retrieve items owned by the user from the database
    items = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user_db.id).offset(skip).limit(
        limit).all()

    if not items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No items found",
            headers={"WWW-Authenticate": "Bearer"},
        )

    # Retrieve tables associated with the items from the database
    dict_tables = {}
    for item in items:
        table = get_dataframe_from_database(item.namedb)
        dict_tables[item.namedb] = table

    # Convert the tables to JSON
    list_json = []
    for name, table in dict_tables.items():
        table_dict_one = {}
        table_json = table.to_json(orient='split')
        table_dict_one["name"] = name
        table_dict_one["data"] = table_json
        list_json.append(table_dict_one)

    # Create a JSON response
    response = JSONResponse(content=list_json)
    return response


# Endpoint to delete an item owned by the current user
@router.delete("/users/me/item/{table_name}")
async def delete_own_item(
        db: Session = Depends(get_db),
        current_user: UserInDB = Depends(get_current_active_user),
        table_name: str = None
):
    """
    Endpoint to delete an item owned by the current authenticated user.

    Args:
        db (Session): Database session dependency.
        current_user (UserInDB): Current authenticated user dependency.
        table_name (str): Name of the table/item to delete.

    Returns:
        dict: Response indicating success or failure.

    Raises:
        HTTPException: Raised for various HTTP status codes with corresponding details.

    Notes:
        - Deletes the table/item associated with the provided name from the database.
    """
    # Get the current user from the database
    current_user_db = db.query(UserTable).filter(UserTable.username == current_user.username).first()

    # Retrieve the item from the database
    item = db.query(UploadedItemTable).filter(UploadedItemTable.owner_id == current_user_db.id).filter(
        UploadedItemTable.namedb == table_name).first()

    if not item:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="Item not found",
            headers={"WWW-Authenticate": "Bearer"},
        )
    else:
        # Delete the table/item from the database
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


# Endpoint to get information about the current user
@router.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_user)):
    """
    Endpoint to get information about the current authenticated user.

    Args:
        current_user (User): Current authenticated user dependency.

    Returns:
        User: Response containing information about the current user.
    """
    return current_user
