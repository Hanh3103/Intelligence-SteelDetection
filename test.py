import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from functions import process_dataframe, apply_pca_with_scaling, apply_kmeans_clustering


class TestProcessDataFrame(unittest.TestCase):

    def setUp(self):
        """Initial setup for tests."""
        self.data = {
            'Date': ['2023-01-01', '2023-01-02', '2023-01-03'],
            'Grade': ['A', 'B', 'C'],
            'Score': [90, 85, 95],
            'Score2': [80, 75, 85],
            'Remarks': ['Good', 'Better', '<Invalid>']
        }
        self.df = pd.DataFrame(self.data)

    def test_valid_input(self):
        """Test with valid input."""
        processed_df = process_dataframe(self.df)
        self.assertIn('Score2', processed_df.columns)
        self.assertNotIn('Remarks', processed_df.columns)  # 'Remarks' should be dropped

    def test_invalid_input_type(self):
        """Test with invalid input type."""
        with self.assertRaises(ValueError):
            process_dataframe("not a dataframe")

    def test_empty_dataframe(self):
        """Test with an empty dataframe."""
        empty_df = pd.DataFrame()
        processed_df = process_dataframe(empty_df)
        self.assertTrue(processed_df.empty)

    def test_no_grade_column(self):
        """Test dataframe without 'Grade' column."""
        df_no_grade = self.df.drop(columns=['Grade'])
        with self.assertRaises(ValueError) as cm:
            process_dataframe(df_no_grade)
        the_exception = cm.exception
        self.assertEqual(str(the_exception), "Input DataFrame does not contain column 'Grade'.")


if __name__ == '__main__':
    unittest.main()


class TestApplyPCAScaling(unittest.TestCase):

    def setUp(self):
        """Initial setup for tests."""
        self.data = {
            'Feature1': np.random.rand(10),
            'Feature2': np.random.rand(10),
            'NonNumeric': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        }
        self.df = pd.DataFrame(self.data)

    def test_with_valid_input_and_scaler(self):
        """Test with valid input and a scaler."""
        scaler = StandardScaler()
        pca_model = apply_pca_with_scaling(self.df, scaler)
        self.assertIsInstance(pca_model, PCA)

    def test_with_valid_input_no_scaler(self):
        """Test with valid input and no scaler."""
        pca_model = apply_pca_with_scaling(self.df)
        self.assertIsInstance(pca_model, PCA)

    def test_with_invalid_input_type(self):
        """Test with invalid input type."""
        with self.assertRaises(ValueError):
            apply_pca_with_scaling("not a dataframe")

    def test_with_empty_dataframe(self):
        """Test with an empty dataframe."""
        empty_df = pd.DataFrame()
        with self.assertRaises(ValueError):
            apply_pca_with_scaling(empty_df)

    def test_with_no_numeric_columns(self):
        """Test with a dataframe with no numeric columns."""
        df_no_numeric = self.df.drop(columns=['Feature1', 'Feature2'])
        with self.assertRaises(ValueError):
            apply_pca_with_scaling(df_no_numeric)


if __name__ == '__main__':
    unittest.main()


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
