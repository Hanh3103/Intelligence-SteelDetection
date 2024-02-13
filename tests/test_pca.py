import unittest
import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from functions import apply_pca_with_scaling


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