from io import StringIO
import pandas as pd
from fastapi.responses import JSONResponse
from fastapi import FastAPI, UploadFile, File, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer, OAuth2PasswordRequestForm
from pydantic import BaseModel
from datetime import datetime, timedelta
from jose import JWTError, jwt
from passlib.context import CryptContext

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

SECRET_KEY= "21dca9fc3d57116f9914aabd3930c325e15be16da134db65ec2d1e9cf6e591ba"
ALGORITHM = "HS256"
ACCESS_TOKEN_EXPIRE_MINUTES = 60


# Initiales Passwort: hausmaus
# Testbar mit swagger
fake_db = {
    "allan": {
    "username": "allan",
    "full_name": "Allan Penkhasov",
    "email": "allan2000@hotmail.de",
    "hashed_password": "$2b$12$.DvL8xf7PlzI3TYL/tfMQejN5LFbYojhCnkN3s/cxSSbLJ3R6iRaa",
    "disabled": False
    }
}

class Token(BaseModel):
    access_token: str
    token_type: str

class TokenData(BaseModel):
    username: str or None = None

class User(BaseModel):
    username: str
    email: str or None = None
    full_name: str or None = None
    disabled: bool or None = None

class UserInDB(User):
    hashed_password: str

pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
oauth_2_scheme = OAuth2PasswordBearer(tokenUrl="token")

def verify_password(plain_password, hashed_password):
    return pwd_context.verify(plain_password, hashed_password)

def get_password_hash(password):
    return pwd_context.hash(password)

def get_user(fake_db, username: str):
    if username in fake_db:
        user_data = fake_db[username]
        return UserInDB(**user_data)

def authenticate_user(fake_db, username: str, password: str):
    user = get_user(fake_db, username)
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

async def get_current_user(token: str = Depends(oauth_2_scheme)):
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
    
    user = get_user(fake_db, token_data.username)
    if user is None:
        raise credentialException
    return user

async def get_current_active_user(current_user: UserInDB = Depends(get_current_user)): 
    if current_user.disabled:
        raise HTTPException(status_code=400, detail="Inactive user")
    return current_user

@fast_app.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(fake_db, form_data.username, form_data.password)
    if not user:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED,
                            detail="Incorrect username or password",
                            headers={"WWW-Authenticate": "Bearer"},)
    access_token_expires = timedelta(ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}

@fast_app.get("/users/me", response_model=User)
async def read_users_me(current_user: User = Depends(get_current_active_user)):
    return current_user

@fast_app.get("/users/me/items", response_model=User)
async def read_own_items(current_user: User = Depends(get_current_active_user)):
    return {"item_name": "test"}


