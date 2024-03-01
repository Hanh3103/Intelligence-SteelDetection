from fastapi.security import OAuth2PasswordBearer
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker, Session
from sqlalchemy.exc import SQLAlchemyError

from db.dbModels import Base
from utils import readConfig

# Read database URL from configuration
url = readConfig.read_config_value("DATABASE", "DB_URL")

# Create a database engine
try:
    engine = create_engine(url, pool_pre_ping=True)
except SQLAlchemyError as e:
    raise Exception(f"Error creating database engine: {str(e)}")

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)

# Create database tables if they do not exist
try:
    Base.metadata.create_all(bind=engine)
except SQLAlchemyError as e:
    raise Exception(f"Error creating database tables: {str(e)}")

# Create an OAuth2PasswordBearer for token authentication
oauth2_scheme = OAuth2PasswordBearer(tokenUrl="token")


def get_current_active_db_session() -> Session:
    """
    Get a database session.

    Yields:
        Session: Database session.

    Usage:
        with get_db() as db:
            # Perform database operations
    """
    try:
        db = SessionLocal()
        yield db
    finally:
        try:
            db.close()
        except SQLAlchemyError as e:
            raise Exception(f"Error closing database session: {str(e)}")


def get_current_active_engine() -> engine:
    try:
        return engine
    except SQLAlchemyError as e:
        raise Exception(f"Error getting database engine: {str(e)}")
