from pandasgui import show

from db.dbModels import UserTable, UploadedItemTable


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


def show_dataframes(dict_dataframes):  # show dataframes in gui but not elegant!
    show(**dict_dataframes)
