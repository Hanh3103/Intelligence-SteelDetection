from io import BytesIO
import pandas as pd
from fastapi import FastAPI, UploadFile, File

import functions

fast_app = FastAPI(openapi_prefix="")


@fast_app.get("/")
async def root():
    return {"message": "Hello World"}


@fast_app.post("/predict_supervised")
async def process_data(dataframe_input: dict):
    df_json = dataframe_input.get("data")
    df = pd.read_json(df_json, orient='split')
    print(df)
    functions.linreg(df)


@fast_app.post("/predict_unsupervised")
async def process_data(file: UploadFile = File(...)):
    df = pd.read_csv(BytesIO(await file.read()))
    print(df)
    functions.linreg(df)
