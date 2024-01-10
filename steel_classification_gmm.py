import numpy as np
import pandas as pd
from sklearn.mixture import GaussianMixture
import json

def recommend_gmm_components(data, max_components=10, random_state=0):
    """
      Recommends the number of Gaussian Mixture Model (GMM) components based on the Bayesian Information Criterion (BIC).

      Parameters:
          data (pd.DataFrame): Input dataset.
          max_components (int): Maximum number of GMM components to be tested.
          random_state (int): Random state for reproducibility.

      Returns:
          int: Recommended number of GMM components.
      """
    bic_values = []
    for n_components in range(1, max_components + 1):
        gmm = GaussianMixture(n_components=n_components, random_state=random_state)
        gmm.fit(data)
        bic_values.append(gmm.bic(data))

    # Find the number of components with the lowest BIC value
    recommended_components = np.argmin(bic_values) + 1
    return recommended_components

def classify_steel_measurements(data, columns_to_cluster, n_components, random_state=0):
    """
       Classifies steel measurements using the Gaussian Mixture Model (GMM) algorithm and returns the results in JSON format.

       Parameters:
           data (pd.DataFrame): Input dataset.
           columns_to_cluster (list): List of columns to be used for clustering.
           n_components (int): Number of GMM components.
           random_state (int): Random state for reproducibility.

       Returns:
           str: JSON format of the GMM results.
       """
    # Extract relevant data
    data_to_cluster = data[columns_to_cluster]

    # Create and fit GMM model
    gmm = GaussianMixture(n_components=n_components, random_state=random_state)
    clusters_gmm = gmm.fit_predict(data_to_cluster)

    # Add GMM clusters to the data
    data['GMM_Cluster'] = clusters_gmm

    # Create JSON format for the results
    result_json = data[['GMM_Cluster']].to_json(orient='records')

    return result_json


csv_file_path = 'C:/Users/Mert/OneDrive/Desktop/FoPra/data/cleaned data/combined_OES_data.csv'
df = pd.read_csv(csv_file_path)

#Beispielaufruf der Funktionen
#df enthält Ihre Daten, columns_to_cluster ist die Liste der zu clusternden Spalten
columns_to_cluster = ['Fe', 'C', 'Si', 'Mn', 'P', 'S', 'Cr', 'Mo', 'Ni', 'Al', 'Co', 'Cu', 'Nb', 'Ti', 'V', 'W', 'Pb', 'Sn', 'B', 'Ca', 'Zr', 'As', 'Bi']
n_components_recommendation = recommend_gmm_components(df[columns_to_cluster])
result_json = classify_steel_measurements(df, columns_to_cluster, n_components_recommendation)

# Ausgabe des JSON-Formats
print(result_json)