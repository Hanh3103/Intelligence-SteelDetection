import pandas as pd
from fastapi import FastAPI
from io import StringIO
import functions
from io import StringIO
from fastapi.responses import JSONResponse
from fastapi import FastAPI
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

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

    # Assuming 'Grade' is the target variable
    target_variable = 'Grade'

    # Extract features and target variable
    x = df.drop(columns=[target_variable])
    y = df[target_variable]

    # Split the dataset into training and testing sets
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # Initialize a RandomForestClassifier (you can choose another algorithm if you prefer)
    model = RandomForestClassifier()

    # Train the model
    model.fit(x_train, y_train)


    # Make predictions on the test set
    predictions = model.predict(x_test)

    response_data = {
        "input_value": dataframe_input,
        "prediction": predictions
    }
    return JSONResponse(content=response_data)

    # 1.) To be implemented: ML Algorithm on Dataframe

    # 2.) To be implemented: return result form Algorithm
