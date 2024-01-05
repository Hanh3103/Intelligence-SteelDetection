import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, DateTime, ForeignKey
from sqlalchemy.engine import URL
from sqlalchemy.orm import declarative_base, sessionmaker, Session
from api import readConfig
from api.models import Base

url = readConfig.read_config_value("DATABASE", "DB_URL")
engine = create_engine(url)
engine.connect()

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
Base.metadata.create_all(bind=engine)


def save_dataframe_to_database(df_in, table_name_db):
    # Save DataFrame to PostgresSQL table
    df_in.to_sql(table_name_db, engine, if_exists="replace", index=False)

    # Close the connection
    # engine_created.dispose()


def get_dataframe_from_database(table_name_db):
    # Load DataFrame data
    df_get = pd.read_sql_table(table_name_db, engine)

    # Close the connection
    # engine_created.dispose()

    return df_get


def delete_table_from_database(table_name_db):
    try:
        metadata = MetaData()
        table_to_delete = Table(table_name_db, metadata, autoload_with=engine)
        table_to_delete.drop(engine)
    except Exception as e:
        print(e)
        print("Table does not exist")

    # Close the connection
    # engine_created.dispose()

# if __name__ == '__main__':
# delete_table_from_database("data_table")
