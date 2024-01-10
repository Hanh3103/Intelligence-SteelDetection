from datetime import datetime
from typing import Optional

from fastapi.security import OAuth2PasswordBearer
from jose import jwt, JWTError
from passlib.context import CryptContext
from sqlalchemy import create_engine, MetaData, Table
from sqlalchemy.orm import sessionmaker

from api.fastAPI import *  # Importing FastAPI related components
from db.dbModels import Base, UserInDB, UserTable, TokenData
from utils import readConfig  # Importing a utility function for reading configuration

# Read database URL from configuration
url = readConfig.read_config_value("DATABASE", "DB_URL")

# Create a database engine
engine = create_engine(url, pool_pre_ping=True)
SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create database tables if they do not exist
Base.metadata.create_all(bind=engine)

# Create a password context for hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")

# Create an OAuth2PasswordBearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")

# Read configuration values
SECRET_KEY = readConfig.read_config_value('TOKEN', 'SECRET_KEY')
ALGORITHM = readConfig.read_config_value('TOKEN', 'ALGORITHM')
ACCESS_TOKEN_EXPIRE_MINUTES = int(readConfig.read_config_value('TOKEN', 'ACCESS_TOKEN_EXPIRE_MINUTES'))


def get_db() -> Session:
    db = SessionLocal()
    try:
        yield db
    finally:
        db.close()


# Function to save a DataFrame to a PostgreSQL table
def save_dataframe_to_database(df_in, table_name_db):
    df_in.to_sql(table_name_db, engine, if_exists="replace", index=False)


# Function to retrieve a DataFrame from a database table
def get_dataframe_from_database(table_name_db):
    return pd.read_sql_table(table_name_db, engine)


# Function to delete a table from the database
def delete_table_from_database(table_name_db):
    metadata = MetaData()
    table_to_delete = Table(table_name_db, metadata, autoload_with=engine, extend_existing=True)
    table_to_delete.drop(engine, checkfirst=True)


# Function to verify a plain password against a hashed password
def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)


# Function to hash a password
def get_password_hash(password):
    return pwd_context.hash(password)


# Function to get a user from the database based on the username
def get_user(username: str, db_session=Depends(SessionLocal)):
    return db_session.query(UserTable).filter(UserTable.username == username).first()


# Function to authenticate a user based on username and password
def authenticate_user(username: str, password: str, db_session=Depends(SessionLocal)):
    user = get_user(username, db_session)
    if not user or not verify_password(password, user.password):
        return False
    return user


# Function to create an access token
def create_access_token(data: dict, expires_delta: Optional[timedelta] = None):
    to_encode = data.copy()
    expire = datetime.utcnow() + (expires_delta or timedelta(minutes=15))
    to_encode.update({"exp": expire})
    return jwt.encode(to_encode, SECRET_KEY, algorithm=ALGORITHM)


# Function to get the current user based on the JWT token
async def get_current_user(token: str = Depends(oauth2_scheme), db_session: Session = Depends(get_db)):
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )

    try:
        # Your JWT decoding logic
        payload = jwt.decode(token, SECRET_KEY, algorithms=[ALGORITHM])
        username: str = payload.get("sub")
        if username is None:
            raise credentials_exception
        token_data = TokenData(username=username)
    except JWTError:
        raise credentials_exception

    # Assuming get_user returns a dictionary with the expected structure
    user_dict = get_user(token_data.username, db_session)
    if user_dict is None:
        raise credentials_exception

    return user_dict


# Function to get the current active user
async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)):
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user


# Function to check if a username is unique in the database
def check_if_user_name_unique(username: str, db_session=Depends(SessionLocal)):
    return not db_session.query(UserTable).filter(UserTable.username == username).first()
