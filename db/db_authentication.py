from fastapi import Depends, HTTPException, status
from passlib.context import CryptContext
from sqlalchemy.orm import Session
from sqlalchemy.exc import SQLAlchemyError

from db.db_instance import get_current_active_db_session as get_db
from utils.database_utils import get_user, UserTable

# Create a password context for hashing and verification
pwd_context = CryptContext(schemes=["bcrypt"], deprecated="auto")
email_context = CryptContext(schemes=["bcrypt"], deprecated="auto")


# Function to hash a password
def get_password_hash(password):
    """
    Hash a plain text password.

    Args:
        password (str): Plain text password.

    Returns:
        str: Hashed password.
    """
    try:
        return pwd_context.hash(password)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error hashing password: {str(e)}")


def get_email_hash(email: str):
    """
    Hash an email address.

    Args:
        email (str): Email address.

    Returns:
        str: Hashed email address.
    """
    try:
        return email_context.hash(email)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error hashing email: {str(e)}")


# Function to verify a plain password against a hashed password
def verify_password(plain_password, hashed_password):
    """
    Verify a plain password against a hashed password.

    Args:
        plain_password (str): Plain text password.
        hashed_password (str): Hashed password.

    Returns:
        bool: True if the passwords match, False otherwise.
    """
    try:
        return pwd_context.verify(plain_password, hashed_password)
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Error verifying password: {str(e)}")


# Function to authenticate a user based on username and password
def authenticate_user(username: str, password: str, db_session: Session = Depends(get_db)):
    """
    Authenticate a user based on username and password.

    Args:
        username (str): Username to authenticate.
        password (str): Password for authentication.
        db_session (Session): Database session.

    Returns:
        UserTable or False: User information if authenticated, False otherwise.
    """
    try:
        user = get_user(username, db_session)

        # Check if user exists and passwords match
        if not user or not verify_password(password, user.password):
            return False

        return user
    except SQLAlchemyError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(e)}")


# Function to check if a username is unique in the database
def check_if_user_name_unique(username: str, db_session: Session = Depends(get_db)):
    """
    Check if a username is unique in the database.

    Args:
        username (str): Username to check.
        db_session (Session): Database session.

    Returns:
        bool: True if the username is unique, False otherwise.
    """
    try:
        return not db_session.query(UserTable).filter(UserTable.username == username).first()
    except SQLAlchemyError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(e)}")


def check_if_email_unique(email: str, db_session: Session):
    """
    Check if an email address is unique in the database.

    Args:
        email (str): Email address to check.
        db_session (Session): Database session.

    Returns:
        bool: True if the email address is unique, False otherwise.
    """
    try:
        return not db_session.query(UserTable).filter(UserTable.email == email).first()
    except SQLAlchemyError as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=f"Database error: {str(e)}")
