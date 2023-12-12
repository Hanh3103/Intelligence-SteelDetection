import pandas as pd
from sqlalchemy import create_engine, MetaData, Table, Column, Integer, String, Float, Boolean, DateTime, ForeignKey

# Define connection parameters
conn_string = 'postgresql://admin:pw@localhost:5432/db'


def save_dataframe_to_database(df_in, table_name_db):
    # Create an engine for connecting to the PostgreSQL database
    engine_created = create_engine(conn_string)

    # Save DataFrame to PostgresSQL table
    df_in.to_sql(table_name_db, engine_created, if_exists="replace", index=False)

    # Close the connection
    engine_created.dispose()


def get_dataframe_from_database(table_name_db):
    # Create an engine for connecting to the PostgreSQL database
    engine_created = create_engine(conn_string)

    # Load DataFrame data
    df_get = pd.read_sql_table(table_name_db, engine_created)

    # Close the connection
    engine_created.dispose()

    return df_get


def delete_table_from_database(table_name_db):
    # Create an engine for connecting to the PostgreSQL database
    engine_created = create_engine(conn_string)

    try:
        metadata = MetaData()
        table_to_delete = Table(table_name_db, metadata, autoload_with=engine_created)
        table_to_delete.drop(engine_created)
    except Exception as e:
        print(e)
        print("Table does not exist")

    # Close the connection
    engine_created.dispose()

if __name__ == '__main__':
    delete_table_from_database("data_table")
