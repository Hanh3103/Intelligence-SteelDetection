# gui.py
from fastapi import FastAPI, File, UploadFile
import pandas as pd
import matplotlib.pyplot as plt
from io import BytesIO

app = FastAPI()


@app.post("/process_excel")
async def process_excel(file: UploadFile = File(...)):

    df = pd.read_excel(BytesIO(await file.read()))

    # PLACE ML CODE HERE

    # Plot the data and save the plot to a BytesIO object
    plt.plot(df['value'], label='Original Data')
    plt.plot(df['prediction'], label='Model Prediction')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Model Prediction vs Original Data')

    plot_img = BytesIO()
    plt.savefig(plot_img, format='png')
    plt.close()

    # Return the plot as a downloadable file
    plot_img.seek(0)
    return {"message": "Processing complete", "plot": plot_img}
