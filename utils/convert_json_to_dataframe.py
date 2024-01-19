from io import StringIO

import pandas as pd


async def convert_json_to_df(df_json):
    df_buffer = StringIO(df_json)
    df = pd.read_json(df_buffer, orient='split')
    return df
