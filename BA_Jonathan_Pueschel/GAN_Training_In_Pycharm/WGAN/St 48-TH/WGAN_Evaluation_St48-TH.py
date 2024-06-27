import keras.src.saving
import pandas as pd
import numpy as np



gen = keras.models.load_model("WGAN_generator_St37_2500Epochs.keras")
lat_noise = np.random.normal(0, 1, (100, 20))
pred = gen.predict(lat_noise)
df = pd.DataFrame(pred, columns=['C', 'Si', 'Mn', "P", "S", "Cr", "Ni", "Al", "Cu"])
print(df.head())

column_range_C = df["C"].min(), df["C"].max()
column_range_Si = df["Si"].min(), df["Si"].max()
column_range_Mn = df["Mn"].min(), df["Mn"].max()
column_range_P = df["P"].min(), df["P"].max()
column_range_S = df["S"].min(), df["S"].max()
column_range_Cr = df["Cr"].min(), df["Cr"].max()
column_range_Ni = df["Ni"].min(), df["Ni"].max()
column_range_Al = df["Al"].min(), df["Al"].max()
column_range_Cu = df["Cu"].min(), df["Cu"].max()

print("Range of C: ", column_range_C)
print("Range of Si: ", column_range_Si)
print("Range of Mn: ", column_range_Mn)
print("Range of P: ", column_range_P)
print("Range of S: ", column_range_S)
print("Range of Cr: ", column_range_Cr)
print("Range of Ni: ", column_range_Ni)
print("Range of Al: ", column_range_Al)
print("Range of Cu: ", column_range_Cu)






















