from io import BytesIO
import pandas as pd
from fastapi import FastAPI, UploadFile, File
from io import StringIO
import functions
from pydantic import BaseModel


class DataModel(BaseModel):
    Date: float
    Program_Name: float
    Grade: float
    Mode: float
    Burn: float
    Fe: float
    C: float
    Si: float
    Mn: float
    P: float
    S: float
    Cr: float
    Mo: float
    Ni: float
    Al: float
    Co: float
    Cu: float
    Nb: float
    Ti: float
    V: float
    W: float
    Pb: float
    Sn: float
    B: float
    Ca: float
    Zr: float
    As: float
    Bi: float
    C_lesser: float
    C_greater: float
    Mn_lesser: float
    Mn_greater: float
    P_lesser: float
    P_greater: float
    S_lesser: float
    S_greater: float
    Cr_lesser: float
    Cr_greater: float
    Mo_lesser: float
    Mo_greater: float
    Al_lesser: float
    Al_greater: float
    Nb_lesser: float
    Nb_greater: float
    Ti_lesser: float
    Ti_greater: float
    V_lesser: float
    V_greater: float
    W_lesser: float
    W_greater: float
    Pb_lesser: float
    Pb_greater: float
    B_lesser: float
    B_greater: float
    Ca_lesser: float
    Ca_greater: float
    Zr_lesser: float
    Zr_greater: float
    As_lesser: float
    As_greater: float
    Bi_lesser: float
    Bi_greater: float
    Ni_lesser: float
    Ni_greater: float
    Unnamed_65_lesser: float
    Unnamed_65_greater: float
    Unnamed_67_lesser: float
    Unnamed_67_greater: float


fast_app = FastAPI(openapi_prefix="")


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
    print(df)
    functions.linreg(df)


@fast_app.post("/predict_unsupervised")
async def process_data(dataframe_input: DataModel):
    df_json_data = dataframe_input.data
    df = pd.DataFrame(df_json_data, columns=dataframe_input.columns)
    functions.linreg(df)
