from pydantic import BaseModel
from sqlalchemy import Column, String, Boolean, Integer, ForeignKey
from sqlalchemy.orm import declarative_base, relationship

Base = declarative_base()


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
    password: str


class UploadedItem(BaseModel):
    id: int
    owner_id: int
    filename: str
    filesize: int
    filepath: str


class UserTable(Base):
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
    __tablename__ = 'uploaded_items'

    id = Column(Integer, primary_key=True, autoincrement=True)
    owner_id = Column(Integer, ForeignKey('users_table.id'), nullable=False)
    filename = Column(String)
    #filetype = Column(String)
    filesize = Column(Integer)
    filepath = Column(String)
    owner = relationship("UserTable", back_populates="items")


