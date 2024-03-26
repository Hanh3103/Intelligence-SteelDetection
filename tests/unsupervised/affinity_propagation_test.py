import pandas as pd
import numpy as np
import unittest
from MLModels.unsupervised.cluster_Algorithm import affinity_propagation

class TestAffinityPropagation(unittest.TestCase):
        def setUp(self):
            # Create sample data for testing
            self.data = pd.DataFrame({
                'A': [1, 2, 3],
                'B': [4, 5, 6]
            })

        def test_valid_input(self):
            # Test valid input data
            result = affinity_propagation(self.data)
            self.assertTrue('Cluster' in result.columns)

        def test_invalid_input(self):
            # Test invalid input data (not a DataFrame)
            with self.assertRaises(ValueError):
                affinity_propagation("invalid_data")

        def test_empty_input(self):
            # Test empty input DataFrame
            empty_data = pd.DataFrame()
            with self.assertRaises(KeyError):
                affinity_propagation(empty_data)

        def test_no_numeric_columns(self):
            # Test DataFrame with no numeric columns
            data_no_numeric = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['x', 'y', 'z']})
            with self.assertRaises(KeyError):
                affinity_propagation(data_no_numeric)

if __name__ == '__main__':
    unittest.main()