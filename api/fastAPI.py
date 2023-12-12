from io import StringIO
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import FastAPI, File, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from datetime import datetime, timedelta
from jose import jwt, JWTError
from passlib.context import CryptContext
from models import User, UserInDB, Token, TokenData, UserTable
import db as dataBase
from sqlalchemy.engine import URL
from sqlalchemy.orm import Session
fast_app = FastAPI()


@fast_app.get("/")
async def root():
    return {"message": "Hello World"}


@fast_app.post("/predict_supervised")
async def process_data(dataframe_input: dict):
    data_input = dataframe_input.get("dataframe_input")
    df_json = data_input.get("data")

    # Verwende StringIO, um die JSON-Zeichenfolge zu einem Datei-ähnlichen Puffer zu machen
    df_buffer = StringIO(df_json)

    # Jetzt können Sie df_buffer in ein DataFrame konvertieren
    df = pd.read_json(df_buffer, orient='split')

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


SECRET_KEY = "21dca9fc3d57116f9914aabd3930c325e15be16da134db65ec2d1e9cf6e591ba"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60

def get_db():
    db = dataBase.SessionLocal()
    try:
        yield db
    finally:
        db.close()

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth_2_scheme = OAuth2PasswordBearer(tokenUrl="token")


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
            hashed_password=user.password,
            disabled=user.disabled
        )
    return None


def authenticate_user(username: str, password: str, db: Session = Depends(get_db)):
    user = get_user(username, db)
    if not user:
        return False
    if not verify_password(password, user.hashed_password):
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

async def get_current_user(token: str = Depends(oauth_2_scheme), db: Session = Depends(get_db)):
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

def checkIfUserNameUnique(username: str, db: Session = Depends(get_db)):
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
                            headers={"WWW-Authenticate": "Bearer"},)
    access_token_expires = timedelta(ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@fast_app.post("/registrate")
async def create_user(newUser: UserInDB, db: Session = Depends(get_db)):

    isUnique = checkIfUserNameUnique(newUser.username, db)

    if isUnique:
        new_user = UserTable(
            username=newUser.username.lower(),
            fullname=newUser.full_name,
            email=newUser.email,
            password=get_password_hash(newUser.hashed_password),
            disabled=newUser.disabled
        )

        db.add(new_user)
        db.commit()
    else:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Username already exists",
                            headers={"WWW-Authenticate": "Bearer"},)
    
@fast_app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@fast_app.get("/users/me/items", response_model=User)
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return current_user
