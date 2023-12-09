import pandas as pd
from fastapi import FastAPI
from io import StringIO
import functions

fast_app = FastAPI()


@fast_app.get("/")
async def root():
    return {"message": "Hello World"}


@fast_app.post("/predict_supervised")
async def process_data(dataframe_input: dict):
    data_input = dataframe_input.get("dataframe_input")
    df_json = data_input.get("data")

    # Verwende StringIO, um die JSON-Zeichenfolge zu einem Datei-ähnlichen Puffer zu machen
    df_buffer = StringIO(df_json)

    # Jetzt können Sie df_buffer in ein DataFrame konvertieren
    df = pd.read_json(df_buffer, orient='split')

    # 1.) To be implemented: ML Algorithm on Dataframe

    # 2.) To be implemented: return result form Algorithm


@fast_app.post("/predict_unsupervised")
async def process_data(dataframe_input: dict):
    data_input = dataframe_input.get("dataframe_input")
    df_json = data_input.get("data")

    df_buffer = StringIO(df_json)

    df = pd.read_json(df_buffer, orient='split')

    # 1.) To be implemented: ML Algorithm on Dataframe

    # 2.) To be implemented: return result form Algorithm
