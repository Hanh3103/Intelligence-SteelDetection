import tkinter as tk
from io import BytesIO
from tkinter import filedialog
import pandas as pd
import requests
from matplotlib import pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split


def upload_file():
    file_path = filedialog.askopenfilename()

    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file type. Please provide a CSV or Excel file.")
        return None

    return df


def predict_data_from_file_unsupervised():
    df = upload_file()

    if df is None:
        return  # Exit function if file upload fails

    df_json = df.to_json(orient='split')

    files = {"dataframe_input": {"data": df_json}}
    response = requests.post("http://127.0.0.1:8000/predict_unsupervised", json=files)
    prediction_result = response.json()
    print(prediction_result)

    img = tk.PhotoImage(data=prediction_result['plot'])
    return img


def predict_data_from_file_supervised():
    df = upload_file()

    df_json = df.to_json(orient='split')

    # Include the DataFrame in the request payload
    files = {"dataframe_input": {"data": df_json}}
    response = requests.post("http://127.0.0.1:8000/predict_supervised", json=files)
    prediction_result = response.json()
    print(prediction_result)
    # img = tk.PhotoImage(data=prediction_result['plot'])
    # return img


"""
def open_sql_dialog():
    sql_string = simpledialog.askstring("Input SQL Statement", "Enter your SQL statement:")
    if sql_string is not None:
        predict_data_from_database(sql_string)


def predict_data_from_database(sql_string):
    print(f"Processing SQL statement: {sql_string}")
"""

"""
def linreg(df):

    # Choose the features and target variable
    features = ['Fe']  # You need to replace 'index' with the actual column name you want to use as a feature
    target = 'Fe'  # You need to replace 'value' with the actual column name you want to predict

    # Split the data into features (X) and target variable (y)
    X = df[features]
    y = df[target]

    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Train the model
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Make predictions
    df['prediction'] = model.predict(X)

    # Plot the data
    plt.plot(df['index'], df['value'], label='Original Data')
    plt.plot(df['index'], df['prediction'], label='Model Prediction')
    plt.legend()
    plt.xlabel('Index')
    plt.ylabel('Value')
    plt.title('Model Prediction vs Original Data')

    # Save the plot to a BytesIO object
    plot_img = BytesIO()
    plt.savefig(plot_img, format='png')
    plt.close()

    # Return the plot as a downloadable file
    plot_img.seek(0)
    return {"message": "Processing complete", "plot": plot_img}
    
    """
