# Import necessary libraries and modules
from datetime import datetime
from typing import Annotated

# Third-party imports
from jose import jwt, JWTError
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

# Local imports
from api.fastAPI import *  # Importing FastAPI related components
from db.dbModels import Base, TokenData, UserTable  # Importing database models and session function
from utils import readConfig  # Importing a utility function for reading configuration

# Read database URL from configuration
url = readConfig.read_config_value("DATABASE", "DB_URL")

# Create a database engine
engine = create_engine(url)
engine.connect()

# Create a session factory
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create database tables if they do not exist
Base.metadata.create_all(bind=engine)


# Function to save a DataFrame to a PostgreSQL table
def save_dataframe_to_database(df_in, table_name_db):
    df_in.to_sql(table_name_db, engine, if_exists="replace", index=False)


# Function to retrieve a DataFrame from a database table
def get_dataframe_from_database(table_name_db):
    df_get = pd.read_sql_table(table_name_db, engine)
    return df_get


# Function to delete a table from the database
def delete_table_from_database(table_name_db):
    try:
        metadata = MetaData()
        table_to_delete = Table(table_name_db, metadata, autoload_with=engine)
        table_to_delete.drop(engine)
    except Exception as e:
        print(e)
        print("Table does not exist")


# Function to verify a plain password against a hashed password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Function to hash a password
def get_password_hash(password):
    return pwd_context.hash(password)


# Function to get a user from the database based on the username
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


# Function to authenticate a user based on username and password
def authenticate_user(username: str, password: str, db: Session = Depends(get_db)):
    user = get_user(username, db)
    if not user:
        return False
    if not verify_password(password, user.password):
        return False
    return user


# Function to create an access token
def create_access_token(data: dict, expires_delta: timedelta or None = None):
    to_encode = data.copy()
    if expires_delta:
        expire = datetime.utcnow() + expires_delta
    else:
        expire = datetime.utcnow + timedelta(minutes=15)

    to_encode.update({"exp": expire})
    encoded_jwt = jwt.encode(to_encode, SECRET_KEY, ALGORITHM)
    return encoded_jwt


# Function to get the current user based on the JWT token
async def get_current_user(token: Annotated[str, Depends(oauth_2_scheme)], db: Session = Depends(get_db)):
    credentialException = HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                                        detail="Could not validate credentials",
                                        headers={"WWW-Authenticate": "Bearer"})

    try:
        # Decode the JWT token
        payload = jwt.decode(token, SECRET_KEY, ALGORITHM)
        username: str = payload.get("sub")
        if username is None:
            raise credentialException
        token_data = TokenData(username=username)
    except JWTError:
        raise credentialException

    # Retrieve user information from the database
    user = get_user(token_data.username, db)
    if user is None:
        raise credentialException
    return user


# Function to get the current active user
async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Function to check if a username is unique in the database
def check_if_user_name_unique(username: str, db: Session = Depends(get_db)):
    user = db.query(UserTable).filter(UserTable.username == username).first()
    return not user  # Return True if the user does not exist, indicating uniqueness
