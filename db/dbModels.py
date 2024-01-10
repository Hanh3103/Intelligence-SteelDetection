# Import necessary modules and classes
from pydantic import BaseModel
from sqlalchemy import Column, String, Boolean, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, relationship
from typing import Dict

# Create a declarative base for SQLAlchemy
Base = declarative_base()


# Define a Pydantic BaseModel for the Token
class Token(BaseModel):
    access_token: str
    token_type: str


# Define a Pydantic BaseModel for TokenData
class TokenData(BaseModel):
    username: str or None = None


# Define a Pydantic BaseModel for User
class User(BaseModel):
    username: str
    email: str or None = None
    full_name: str or None = None
    disabled: bool or None = None


# Define a Pydantic BaseModel for UserInDB, inheriting from User
class UserInDB(BaseModel):
    username: str
    full_name: str
    email: str
    password: str
    disabled: bool


# Define a Pydantic BaseModel for UploadedItem
class UploadedItem(BaseModel):
    filename: str
    filesize: int
    data: dict


# Define a SQLAlchemy Table for UserTable
class UserTable(Base):
    __tablename__ = 'users_table'

    # Define columns for the table
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String)
    fullname = Column(String)
    email = Column(String, unique=True)
    password = Column(String)
    disabled = Column(Boolean)

    # Define a relationship with UploadedItemTable
    items = relationship("UploadedItemTable",
                         back_populates="owner",
                         cascade="all, delete",
                         uselist=True)


# Define a SQLAlchemy Table for UploadedItemTable
class UploadedItemTable(Base):
    __tablename__ = 'uploaded_items'

    # Define columns for the table
    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_id = Column(Integer, ForeignKey('users_table.id'), nullable=False)
    filename = Column(String)
    # Add a commented-out column for filetype
    # filetype = Column(String)
    filesize = Column(Integer)
    namedb = Column(String)

    # Define a relationship with UserTable
    owner = relationship("UserTable", back_populates="items")
