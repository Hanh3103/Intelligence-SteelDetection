import pandas as pd
import numpy as np
import unittest
from MLModels.unsupervised.cluster_Algorithm import tsne_algorithm
from sklearn.preprocessing import FunctionTransformer
from sklearn.manifold import TSNE

class TestTSNEAlgorithm(unittest.TestCase):
    def setUp(self):
        # Create sample data for testing
        self.data = pd.DataFrame({
            'A': [1, 2, 3],
            'B': [4, 5, 6],
            'C': [7, 8, 9]
        })

    def test_2D_mode(self):
        # Test t-SNE algorithm with 2D mode
        transformed_data = tsne_algorithm(self.data, perplexity=2, mode='2D', steps=1000)
        self.assertEqual(len(transformed_data), len(self.data))
        self.assertEqual(len(transformed_data[0]), 2)

    def test_3D_mode(self):
        # Test t-SNE algorithm with 3D mode
        transformed_data = tsne_algorithm(self.data, perplexity=2, mode='3D', steps=1000)
        self.assertEqual(len(transformed_data), len(self.data))
        self.assertEqual(len(transformed_data[0]), 3)

    def test_empty_input(self):
        # Test t-SNE algorithm with empty input
        empty_data = pd.DataFrame()
        with self.assertRaises(ValueError):
            tsne_algorithm(empty_data, perplexity=2, mode='2D', steps=1000)

    def test_invalid_mode(self):
        # Test t-SNE algorithm with invalid mode
        with self.assertRaises(ValueError):
            tsne_algorithm(self.data, perplexity=2, mode='4D', steps=1000)

    def test_no_numeric_columns(self):
        # Test t-SNE algorithm with no numeric columns
        data = pd.DataFrame({'A': ['a', 'b', 'c'], 'B': ['d', 'e', 'f']})
        with self.assertRaises(ValueError):
            tsne_algorithm(data, perplexity=2, mode='2D', steps=1000)

    def test_perplexity_greater_than_samples(self):
        # Test t-SNE algorithm with perplexity greater than samples
        with self.assertRaises(ValueError):
            tsne_algorithm(self.data, perplexity=10, mode='2D', steps=1000)

if __name__ == '__main__':
    unittest.main()