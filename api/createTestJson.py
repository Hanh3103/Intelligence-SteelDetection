import json

import pandas as pd

# Create a simple DataFrame
df = pd.DataFrame({
    'A': [1, 2, 3],
    'B': [4, 5, 6]
})

# Convert the DataFrame to a JSON string
df_json = df.to_json(orient='split')

# Define the file size and file name
filesize = len(df_json)
filename = 'test_file.json'

# Create the 'data_frame' dictionary
data_frame = {
    "dataframe_input": {
        "data": df_json,
        "filesize": filesize,
        "filename": filename
    }
}

# Print the 'data_frame' dictionary
print(data_frame)

