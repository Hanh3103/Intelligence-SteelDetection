# Import necessary modules and classes
from pydantic import BaseModel
from sqlalchemy import Column, String, Boolean, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

# Create a declarative base for SQLAlchemy
Base = declarative_base()


class Token(BaseModel):
    """
    Pydantic BaseModel for representing a Token.

    Attributes:
        - `access_token` (str): The access token.
        - `token_type` (str): The type of the token.
    """
    access_token: str
    token_type: str


class TokenData(BaseModel):
    """
    Pydantic BaseModel for representing TokenData.

    Attributes:
        - `username` (str or None): The username associated with the token data.
    """
    username: str or None = None


class User(BaseModel):
    """
    Pydantic BaseModel for representing a User.

    Attributes:
        - `username` (str): The username of the user.
        - `email` (str or None): The email of the user (optional).
        - `full_name` (str or None): The full name of the user (optional).
        - `disabled` (bool or None): Whether the user is disabled (optional).
    """
    username: str
    email: str or None = None
    full_name: str or None = None
    disabled: bool or None = None


class UserInDB(BaseModel):
    """
    Pydantic BaseModel for representing UserInDB, inheriting from User.

    Attributes:
        - `username` (str): The username of the user.
        - `full_name` (str): The full name of the user.
        - `email` (str): The email of the user.
        - `password` (str): The password of the user.
        - `disabled` (bool): Whether the user is disabled.
    """
    username: str
    full_name: str
    email: str
    password: str
    disabled: bool


class UploadedItem(BaseModel):
    """
    Pydantic BaseModel for representing an UploadedItem.

    Attributes:
        - `filename` (str): The filename of the uploaded item.
        - `filesize` (int): The filesize of the uploaded item.
        - `data` (dict): The data associated with the uploaded item.
    """
    filename: str
    filesize: int
    data: dict


class UserTable(Base):
    """
    SQLAlchemy Table for representing UserTable.

    Columns:
        - `id` (Integer): Primary key for the table.
        - `username` (String): Username of the user.
        - `fullname` (String): Full name of the user.
        - `email` (String): Email of the user.
        - `password` (String): Password of the user.
        - `disabled` (Boolean): Whether the user is disabled.
    """
    __tablename__ = 'users_table'

    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String)
    fullname = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    disabled = Column(Boolean)

    items = relationship("UploadedItemTable",
                         back_populates="owner",
                         cascade="all, delete",
                         uselist=True)


class UploadedItemTable(Base):
    """
    SQLAlchemy Table for representing UploadedItemTable.

    Columns:
        - `id` (Integer): Primary key for the table.
        - `owner_id` (Integer): Foreign key referencing UserTable.
        - `filename` (String): Filename of the uploaded item.
        - `filesize` (Integer): Filesize of the uploaded item.
        - `namedb` (String): Namedb associated with the uploaded item.
    """
    __tablename__ = 'uploaded_items'

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_id = Column(Integer, ForeignKey('users_table.id'), nullable=False)
    filename = Column(String)
    filesize = Column(Integer)
    namedb = Column(String)

    owner = relationship("UserTable", back_populates="items")
