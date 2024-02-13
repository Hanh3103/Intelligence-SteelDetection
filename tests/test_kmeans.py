import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from functions import apply_kmeans_clustering


class TestApplyKMeansClustering(unittest.TestCase):

    def setUp(self):
        """Initial setup for tests."""
        self.data = {
            'Feature1': np.random.rand(10),
            'Feature2': np.random.rand(10)
        }
        self.df = pd.DataFrame(self.data)

    def test_valid_input_without_scaler(self):
        """Test with valid input without a scaler."""
        clustered_df = apply_kmeans_clustering(self.df, k=3)
        self.assertIn('Cluster', clustered_df.columns)

    def test_valid_input_with_scaler(self):
        """Test with valid input with a scaler."""
        scaler = StandardScaler()
        clustered_df = apply_kmeans_clustering(self.df, k=3, scaler=scaler)
        self.assertIn('Cluster', clustered_df.columns)

    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with self.assertRaises(ValueError):
            apply_kmeans_clustering("not a dataframe", k=3)

    def test_empty_dataframe(self):
        """Test with an empty dataframe."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            apply_kmeans_clustering(empty_df, k=3)

    def test_invalid_k_value(self):
        """Test with invalid k value."""
        with self.assertRaises(ValueError):
            apply_kmeans_clustering(self.df, k=1)  # k must be greater than 1

    def test_no_numeric_columns(self):
        """Test with a dataframe with no numeric columns."""
        df_no_numeric = self.df.drop(columns=['Feature1', 'Feature2'])
        with self.assertRaises(ValueError):
            apply_kmeans_clustering(df_no_numeric, k=3)


if __name__ == '__main__':
    unittest.main()