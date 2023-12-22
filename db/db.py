from datetime import datetime
from typing import Annotated

from jose import jwt, JWTError
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

from api.fastAPI import *
from db.dbModels import Base, TokenData
from utils import readConfig

url = readConfig.read_config_value("DATABASE", "DB_URL")
engine = create_engine(url)
engine.connect()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def save_dataframe_to_database(df_in, table_name_db):
    # Save DataFrame to PostgresSQL table
    df_in.to_sql(table_name_db, engine, if_exists="replace", index=False)

    # Close the connection
    # engine_created.dispose()


def get_dataframe_from_database(table_name_db):
    # Load DataFrame data
    df_get = pd.read_sql_table(table_name_db, engine)

    # Close the connection
    # engine_created.dispose()

    return df_get


def delete_table_from_database(table_name_db):
    try:
        metadata = MetaData()
        table_to_delete = Table(table_name_db, metadata, autoload_with=engine)
        table_to_delete.drop(engine)
    except Exception as e:
        print(e)
        print("Table does not exist")

    # Close the connection
    # engine_created.dispose()


# if __name__ == '__main__':
# delete_table_from_database("data_table")


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
