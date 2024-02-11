import logging
from datetime import timedelta

from fastapi import APIRouter
from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from sqlalchemy.orm import Session
from db.db_instance import get_current_active_db_session as get_db
from utils.database_utils import create_access_token
from db.db_authentication import authenticate_user, get_email_hash, get_password_hash, check_if_user_name_unique ,check_if_email_unique
from db.dbModels import UserInDB, Token, UserTable
from utils.constants import ACCESS_TOKEN_EXPIRE_MINUTES

router = APIRouter()

# Configure logging
logger = logging.getLogger(__name__)


@router.post("/token", response_model=Token)
async def login_for_access_token(form_data: OAuth2PasswordRequestForm = Depends(), db: Session = Depends(get_db)):
    try:
        # Input validation
        if not form_data.username or not form_data.password:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username and password are required",
            )

        # Authentication
        user = authenticate_user(form_data.username.lower(), form_data.password, db)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Incorrect username or password",
                headers={"WWW-Authenticate": "Bearer"},
            )

        # Token generation
        access_token_expires = timedelta(minutes=ACCESS_TOKEN_EXPIRE_MINUTES)
        access_token = create_access_token(data={"sub": user.username}, expires_delta=access_token_expires)

        return {"access_token": access_token, "token_type": "bearer"}

    except Exception as e:
        # Log errors
        logger.error(f"An error occurred: {e}")

        # Raise a generic HTTP 500 error in case of unexpected exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )


@router.post("/registrate")
async def create_user(new_user: UserInDB, db: Session = Depends(get_db)):
    try:
        # Check if the username is unique
        is_unique_username = check_if_user_name_unique(new_user.username, db)
        if not is_unique_username:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Username already exists",
            )

        # Check if the email address is unique
        is_unique_email = check_if_email_unique(new_user.email, db)
        if not is_unique_email:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail="Email address already exists",
            )

        # Create a new user
        user = UserTable(
            username=new_user.username.lower(),
            fullname=new_user.full_name,
            email=get_email_hash(new_user.email),
            password=get_password_hash(new_user.password),
            disabled=new_user.disabled
        )

        # Add the user to the database
        db.add(user)
        db.commit()

        return {"message": "User created successfully"}

    except Exception as e:
        # Log any unexpected errors
        logger.error(f"An error occurred: {e}")

        # Raise a generic HTTP 500 error in case of unexpected exceptions
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail="Internal server error",
        )
