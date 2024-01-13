import unittest
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from cluster_Algorithm import gmm_algorithm, recommend_gmm_components

class TestSteelClassification(unittest.TestCase):

    def setUp(self):
        # Read data from CSV for testing
        csv_file_path = 'C:/Users/Mert/OneDrive/Desktop/FoPra/data/cleaned data/combined_OES_data.csv'
        self.df = pd.read_csv(csv_file_path)

    def test_recommend_gmm_components(self):
        # Test if the recommended number of components is an integer
        # Exclude non-numeric columns from the selection
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        recommended_components = recommend_gmm_components(self.df[numeric_columns])
        self.assertIsInstance(recommended_components, int)

        # Test if the recommended number is within a reasonable range
        self.assertGreaterEqual(recommended_components, 1)
        self.assertLessEqual(recommended_components, 10)



    def test_csv_data_types(self):
        # Test if the data types in the CSV file are as expected
        expected_data_types = {'Fe': np.float64, 'C': np.float64, 'Si': np.float64, 'Mn': np.float64,
                                   'P': np.float64, 'S': np.float64,
                                   'Cr': np.float64, 'Mo': np.float64, 'Ni': np.float64, 'Al': np.float64,
                                   'Co': np.float64, 'Cu': np.float64,
                                   'Nb': np.float64, 'Ti': np.float64, 'V': np.float64, 'W': np.float64,
                                   'Pb': np.float64, 'Sn': np.float64,
                                   'B': np.float64, 'Ca': np.float64, 'Zr': np.float64, 'As': np.float64,
                                   'Bi': np.float64}
        self.assertTrue(all(self.df[column].dtype == expected_data_types[column] for column in expected_data_types))

    def test_recommend_gmm_components_valid_range(self):
         # Test if the recommended number of components is within a reasonable range
         numeric_columns = self.df.select_dtypes(include=[np.number]).columns
         recommended_components = recommend_gmm_components(self.df[numeric_columns])
         self.assertGreaterEqual(recommended_components, 1)
         self.assertLessEqual(recommended_components, 10)







