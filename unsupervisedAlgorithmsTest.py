import json
import unittest

import numpy as np
import pandas as pd
from sklearn.datasets import make_blobs
from sklearn.preprocessing import StandardScaler
from unsupervisedAlgorithms import selectClusteringAlgorithm, selectReductionAlgorithm, selectPCA, selectTSNE, selectUMAP, selectFUZZY
from unsupervisedAlgorithms import modelsClustering, modelsReduction
from sklearn.decomposition import PCA
class TestClusterAndReduction(unittest.TestCase):

    def setUp(self):
        # Create a sample dataframe for testing
        X, _ = make_blobs(n_samples=100, centers=3, random_state=42, cluster_std=2.0)
        self.df = pd.DataFrame(X, columns=['Feature1', 'Feature2'])

    def test_selectClusteringAlgorithm(self):
        # Test KMeans clustering
        result = selectClusteringAlgorithm('KMeans', self.df, k_components=3, scaler=StandardScaler())
        self.assertTrue('Cluster' in result.columns)

        # Test Gaussian Mixture Model clustering
        result = selectClusteringAlgorithm('GaussianMixture', self.df, k_components=3)
        self.assertTrue('Cluster' in result.columns)

        # Test Hierarchical clustering
        result = selectClusteringAlgorithm('Hierarchical', self.df, k_components=3)
        self.assertTrue('Cluster' in result.columns)

        # Test invalid clustering model
        with self.assertRaises(ValueError):
            selectClusteringAlgorithm('Fuzzyy', self.df, k_components=3)

    def test_selectReductionAlgorithm(self):
        # Test PCA reduction
        result = selectReductionAlgorithm(self.df, 'PCA')
        print(result)
        self.assertIsInstance(result,PCA)

        # Test TSNE reduction
        result = selectReductionAlgorithm(self.df, 'Tsne', perplexity=5, mode='2D')
        # Check if the result is a NumPy array
        self.assertTrue('Dimension1' in result)
        self.assertTrue('Dimension2' in result)

        # Check if the NumPy array has the expected shape
        expected_shape = 2
        self.assertEqual(len(json.loads(result)), expected_shape)

        # Test UMAP reduction
        result = selectReductionAlgorithm(self.df, 'Umap', n_neighbors=15, min_dist=0.1, n_components=2)
        self.assertEqual(result.shape[1], 2)  # UMAP reduces to 2 dimensions

        # Test invalid reduction model
        with self.assertRaises(ValueError):
            selectReductionAlgorithm(self.df, 'f', n_neighbors=5, mode='2D')




if __name__ == '__main__':
    unittest.main()
