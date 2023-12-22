from tkinter import filedialog, Tk
from matplotlib import pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import requests


def upload_file():
    # Function to upload a file using tkinter file dialog
    file_path = filedialog.askopenfilename()

    # Extract filename and filesize
    filename = file_path.split('/')[-1]
    filesize = file_path.split('/')[-2]

    # Read the file based on its type (CSV or Excel)
    if file_path.endswith('.csv'):
        df = pd.read_csv(file_path)
    elif file_path.endswith(('.xls', '.xlsx')):
        df = pd.read_excel(file_path)
    else:
        print("Unsupported file type. Please provide a CSV or Excel file.")
        return None

    return df, filesize, filename


def predict_data_from_file_unsupervised():
    # Function to predict data from an uploaded file in unsupervised mode
    df, filesize, filename = upload_file()

    # Convert DataFrame to JSON
    df_json = df.to_json(orient='split')

    # Include the DataFrame in the request payload
    files = {"dataframe_input": {"data": df_json, "filesize": filesize, "filename": filename}}
    response = requests.post("http://127.0.0.1:8000/predict_supervised", json=files)

    # Process the response data
    data = response.json()
    df2 = pd.DataFrame.from_dict(data)
    print(df2)

    # Plot the data using Matplotlib
    plot_frame = Tk()
    plot_frame.title("Matplotlib Plot in Tkinter")

    # Assuming 'Date' is a column in your DataFrame
    x_values = df['Date']
    # Assuming 'Fe' is another column in your DataFrame
    y_values = df['Fe']

    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='Fe over Date')
    ax.set_xlabel('Date')
    ax.set_ylabel('Fe')
    ax.set_title('Matplotlib Plot')

    # Embed the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Display the Tkinter window
    plot_frame.mainloop()

    # To be implemented: plot the response data to the GUI


def predict_data_from_file_supervised():
    # Function to predict data from an uploaded file in supervised mode
    df = upload_file()

    # Convert DataFrame to JSON
    df_json = df.to_json(orient='split')

    # Include the DataFrame in the request payload
    files = {"dataframe_input": {"data": df_json}}
    response = requests.post("http://127.0.0.1:8000/predict_supervised", json=files)

    # Process the response data
    data = response.json()
    df2 = pd.DataFrame.from_dict(data)
    print(df2)

    # Plot the data using Matplotlib
    plot_frame = Tk()
    plot_frame.title("Matplotlib Plot in Tkinter")

    # Assuming 'Date' is a column in your DataFrame
    x_values = df['Date']
    # Assuming 'Fe' is another column in your DataFrame
    y_values = df['Fe']

    # Create a Matplotlib figure
    fig, ax = plt.subplots()
    ax.plot(x_values, y_values, label='Fe over Date')
    ax.set_xlabel('Date')
    ax.set_ylabel('Fe')
    ax.set_title('Matplotlib Plot')

    # Embed the Matplotlib figure in the Tkinter window
    canvas = FigureCanvasTkAgg(fig, master=plot_frame)
    canvas.draw()
    canvas.get_tk_widget().pack()

    # Display the Tkinter window
    plot_frame.mainloop()

    # To be implemented: plot the response data to the GUI


# The commented-out section can be uncommented and implemented as needed.
"""
def open_sql_dialog():
    # Function to open a dialog for entering an SQL statement
    sql_string = simpledialog.askstring("Input SQL Statement", "Enter your SQL statement:")
    if sql_string is not None:
        predict_data_from_database(sql_string)

def predict_data_from_database(sql_string):
    # Function to predict data from a database using an SQL statement
    print(f"Processing SQL statement: {sql_string}")
"""
