import unittest

import pandas as pd
import numpy as np
from MLModels.unsupervised.cluster_Algorithm import kmeans


class TestKMeans(unittest.TestCase):
    def setUp(self):
        # Erstellen eines einfachen DataFrames für Tests
        self.test_df = pd.DataFrame({
            'feature1': [1, 2, 3, 4, 5],
            'feature2': [5, 4, 3, 2, 1]
        })

    def test_valid_input(self):
        # Test mit gültigem DataFrame
        labels = kmeans(self.test_df, k=2)
        self.assertEqual(len(labels), len(self.test_df),
                         "Die Länge der Labels sollte der Anzahl der DataFrame-Zeilen entsprechen.")

    def test_dataframe_without_numeric_columns(self):
        # Testet, ob ein ValueError ausgelöst wird, wenn der DataFrame keine numerischen Spalten enthält
        non_numeric_df = pd.DataFrame({
            'feature1': ['a', 'b', 'c', 'd', 'e'],
            'feature2': ['f', 'g', 'h', 'i', 'j']
        })
        with self.assertRaises(ValueError):
            kmeans(non_numeric_df, k=2)

    def test_return_type(self):
        # Testet den Rückgabetyp
        labels = kmeans(self.test_df, k=2)
        self.assertIsInstance(labels, pd.DataFrame, "Die Rückgabe sollte ein numpy array von Labels sein.")

    def test_invalid_n_clusters(self):
        with self.assertRaises(ValueError):
            kmeans(self.test_df, k=-1)
        with self.assertRaises(ValueError):
            kmeans(self.test_df, k=0)
        with self.assertRaises(ValueError):
            kmeans(self.test_df, k=len(self.test_df) + 1)

    def test_consistency_of_cluster_assignments(self):
        # Erstellen eines DataFrames mit zwei offensichtlichen Clustern
        df_with_clusters = pd.DataFrame({
            'feature1': [1, 2, 2, 3, 11, 12, 12, 13],
            'feature2': [1, 2, 1, 2, 11, 12, 11, 12]
        })
        df_labels = kmeans(df_with_clusters, k=2)
        # Zugreifen auf die Cluster-Labels in der "Cluster"-Spalte
        labels = df_labels['Cluster']

        # Überprüfen, ob alle Punkte eines offensichtlichen Clusters demselben Cluster zugeordnet werden
        cluster_1 = labels.iloc[0]
        for i in range(1, 4):
            self.assertEqual(labels.iloc[i], cluster_1,
                             "Punkte in demselben offensichtlichen Cluster sollten demselben Cluster zugeordnet werden.")
        cluster_2 = labels.iloc[4]
        for i in range(5, 8):
            self.assertEqual(labels.iloc[i], cluster_2,
                             "Punkte in demselben offensichtlichen Cluster sollten demselben Cluster zugeordnet werden.")
        self.assertNotEqual(cluster_1, cluster_2,
                            "Die zwei offensichtlichen Cluster sollten unterschiedlichen Clustern zugeordnet werden.")

    def test_handling_high_dimensionality(self):
        # Erstellen eines DataFrames mit hoher Dimensionalität
        high_dim_df = pd.DataFrame(np.random.rand(100, 50))  # 100 Zeilen, 50 Features
        labels = kmeans(high_dim_df, k=5)
        self.assertEqual(len(labels), 100,
                         "Die Methode sollte in der Lage sein, mit DataFrames hoher Dimensionalität umzugehen.")


if __name__ == '__main__':
    unittest.main()
