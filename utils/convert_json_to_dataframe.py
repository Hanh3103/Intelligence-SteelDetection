from io import StringIO
import pandas as pd
from fastapi import HTTPException, status


async def convert_json_to_df(df_json):
    try:
        # Create a StringIO object to mimic a file-like object from a string
        df_buffer = StringIO(df_json)

        # Read the JSON data from the buffer into a pandas DataFrame
        df = pd.read_json(df_buffer, orient='split')

        # Return the resulting DataFrame
        return df
    except Exception as e:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
                            detail=f"Error converting JSON to DataFrame: {str(e)}")
