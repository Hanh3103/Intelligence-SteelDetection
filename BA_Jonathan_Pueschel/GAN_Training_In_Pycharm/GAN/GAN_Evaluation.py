import keras.src.saving
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from matplotlib.lines import Line2D
# Load Data
data = pd.read_csv("../../labeled_OES_data_2.csv")

# convert steel types to numbers
# Note: in our project steel types are not predefined, but will be extracted from the supplied file of data
classes_dict = {"St 48" : 0, "St 37" : 1, "St Si" : 2, "St 48-TH" : 3, "Puddelstahl" : 4}
data["steel_class"] = [classes_dict[c] for c in data["steel_class"] ]

# make data numeric
for column in data.columns:
        for index, value in data[column].items():
            if isinstance(value, str):
                data.at[index, column] = value.replace(",", ".")
data = data.map(pd.to_numeric)



element_data_grouped_by_steeltype = data.groupby("steel_class")
dfs_by_steeltype = {}

for type, group_data in element_data_grouped_by_steeltype:
    dfs_by_steeltype[type] = group_data.iloc[:, :-1]

for type in dfs_by_steeltype.keys():
    print(f"\nSteel Type {type}:")
    print(dfs_by_steeltype[type].head())



gen = keras.models.load_model("GAN_generator_St37_1000Epochs.keras")
lat_noise = np.random.normal(0, 1, (100, 10))
pred = gen.predict(lat_noise)
df = pd.DataFrame(pred, columns=['C', 'Si', 'Mn', "P", "S", "Cr", "Ni", "Al", "Cu"])
print(df.head())

#Ranges
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


# Distributions
for col in df.columns:  # Loop through columns (excluding the first)
    df[col].plot.hist()
    plt.title(col, fontsize=40)  # Set title as element name
    plt.ylabel("Frequency", fontsize=20)
    plt.xticks(fontsize=15)  # Adjust x-axis tick label font size
    plt.yticks(fontsize=17)
    #plt.savefig(f"Distribution_{col}.png", format='png')
    plt.show()
    plt.close()


# 3D Plot

scaler = StandardScaler()
scaled_augmented_rows = scaler.fit_transform(pred)

pca = PCA(3)
pca_augmented_rows = pca.fit_transform(scaled_augmented_rows)

# Only Augmented Data
x = pca_augmented_rows[:, 0]
y = pca_augmented_rows[:, 1]
z = pca_augmented_rows[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], c='lightgreen', marker='o')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D augmented data St 37')

ax.set_xlim(-2.5, 18)
#ax.set_ylim(-2, 4)
ax.set_zlim(-2, 4)

plt.tight_layout()
#plt.savefig('GAN_Aug_ONLYSt37.png', format='png')

plt.show()


st37_samples = dfs_by_steeltype[1]
scaled_St37_rows = scaler.fit_transform(st37_samples)
pca_St37_rows = pca.fit_transform(scaled_St37_rows)

x_st37 = pca_St37_rows[:, 0]
y_st37 = pca_St37_rows[:, 1]
z_st37 = pca_St37_rows[:, 2]

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for i in range(len(x)):
        ax.scatter(x[i], y[i], z[i], c='lightgreen', marker='o', label="Augmented St 37")

for i in range(len(x_st37)):
        ax.scatter(x_st37[i], y_st37[i], z_st37[i], c='g', marker='o', alpha=0.08, label="Real St 37")

legend_elements = []
legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="lightgreen", markersize=10, label="Augmented St 37"))
legend_elements.append(Line2D([0], [0], marker="o", color="w", markerfacecolor="g", markersize=10, label="Real St 37"))
plt.legend(handles=legend_elements, loc="upper right")

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D augmented data St 37')

ax.set_xlim(-2.5, 18)
#ax.set_ylim(-2, 10)
#ax.set_zlim(-4, 8)

plt.tight_layout()
#plt.savefig('GAN_Aug_St37.png', format='png')

plt.show()


# Outliers
outliers = []
index_outliers = []
for index in range(len(pca_augmented_rows)):
    if pca_augmented_rows[index][0] > 8:
        outliers.append(pca_augmented_rows[index])
        index_outliers.append(index)

fig = plt.figure()
ax = fig.add_subplot(111, projection='3d')

for outlier in outliers:
    ax.scatter(outlier[0], outlier[1], outlier[2], c='r', marker='o')

ax.set_xlabel('Principal Component 1')
ax.set_ylabel('Principal Component 2')
ax.set_zlabel('Principal Component 3')
ax.set_title('3D augmented data St 37')

ax.set_xlim(-2.5, 18)
ax.set_ylim(-2, 4)
ax.set_zlim(-2, 4)

plt.tight_layout()
#plt.savefig('Outliers_Aug_St37.png', format='png')

plt.show()






