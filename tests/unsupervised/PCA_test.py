import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from MLModels.unsupervised.cluster_Algorithm import pca


class TestPCA(unittest.TestCase):
    def setUp(self):
        # Erstellen eines einfachen DataFrames für Tests
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })
        self.scaler = StandardScaler()

    def test_valid_input_without_scaler(self):
        # Test ohne Skalierer
        result = pca(self.test_df)
        self.assertIsNotNone(result, "PCA sollte ein Modell zurückgeben, wenn ein gültiger DataFrame übergeben wird.")

    def test_valid_input_with_scaler(self):
        # Test mit Skalierer
        result = pca(self.test_df, scaler=self.scaler)
        self.assertIsNotNone(result,
                             "PCA sollte ein Modell zurückgeben, wenn ein gültiger DataFrame und ein Skalierer übergeben werden.")

    def test_invalid_input(self):
        # Test mit ungültiger Eingabe
        with self.assertRaises(ValueError):
            pca("kein DataFrame")

    def test_return_type(self):
        # Testet den Rückgabetyp
        result = pca(self.test_df, scaler=self.scaler)
        self.assertIsInstance(result, np.ndarray, "Die Rückgabe sollte ein Array sein.")

    def test_dataframe_without_numeric_columns(self):
        # Erstellen eines DataFrames ohne numerische Spalten
        non_numeric_df = pd.DataFrame({
            'feature1': ['a', 'b', 'c', 'd', 'e'],
            'feature2': ['f', 'g', 'h', 'i', 'j']
        })
        with self.assertRaises(ValueError):
            pca(non_numeric_df)


if __name__ == '__main__':
    unittest.main()
