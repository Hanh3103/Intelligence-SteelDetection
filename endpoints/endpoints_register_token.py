from datetime import timedelta

from fastapi import APIRouter
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session

from db.db import (authenticate_user, create_access_token, check_if_user_name_unique, get_password_hash, \
                   get_db)
from db.dbModels import UserInDB, Token, UserTable
from utils.constants import ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()


# Token endpoint for user authentication
@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    user = authenticate_user(form_data.username.lower(), form_data.password, db)
    if not user:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    access_token_expires = timedelta(ACCESS_TOKEN_EXPIRE_MINUTES)
    access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)
    return {"access_token": access_token, "token_type": "bearer"}


@router.post("/registrate")
async def create_user(new_user: UserInDB, db: Session = Depends(get_db)):
    is_unique = check_if_user_name_unique(new_user.username, db)

    if is_unique:
        new_user = UserTable(
            username=new_user.username.lower(),
            fullname=new_user.full_name,
            email=new_user.email,
            password=get_password_hash(new_user.password),
            disabled=new_user.disabled
        )

        db.add(new_user)
        db.commit()
        return {"message": "User created successfully"}
    else:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Username already exists",
            headers={"WWW-Authenticate": "Bearer"},
        )
