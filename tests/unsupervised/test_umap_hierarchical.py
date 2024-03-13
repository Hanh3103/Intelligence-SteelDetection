import unittest
import pandas as pd
import numpy as np
from MLModels.unsupervised.cluster_Algorithm import apply_umap, hierarchical


class TestUmapAndHierarchical(unittest.TestCase):

    def setUp(self):
        """Initial setup for tests."""
        self.data = {
            'Feature1': np.random.rand(10),
            'Feature2': np.random.rand(10),
            'NonNumeric': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J']
        }
        self.df = pd.DataFrame(self.data)

    def test_apply_umap_input_data_type(self):
        with self.assertRaises(ValueError):
            apply_umap(self.df.values)  # Passing numpy array instead of pandas DataFrame should raise an error

    def test_apply_umap_no_numeric_columns(self):
        with self.assertRaises(ValueError):
            apply_umap(
                pd.DataFrame({'NonNumeric': ['A', 'B']}))  # DataFrame with no numeric columns should raise an error

    def test_hierarchical_input_data_type(self):
        with self.assertRaises(ValueError):
            hierarchical(self.df.values,
                         n_clusters=2)  # Passing numpy array instead of pandas DataFrame should raise an error

    def test_hierarchical_cluster_column_added(self):
        result_df = hierarchical(self.df, n_clusters=2)
        self.assertTrue(
            'Cluster' in result_df.columns)  # Check if the 'Cluster' column is added to the output DataFrame

    def test_apply_umap_output_shape(self):
        u = apply_umap(self.df)
        self.assertEqual(u.shape, (self.df.shape[0], 2))  # Check if the output shape is as expected

    def test_hierarchical_cluster_assignment(self):
        result_df = hierarchical(self.df, n_clusters=2)
        unique_clusters = result_df['Cluster'].unique()
        self.assertEqual(len(unique_clusters), 2)  # Check if the correct number of clusters are assigned

    def test_hierarchical_cluster_labels(self):
        result_df = hierarchical(self.df, n_clusters=2)
        cluster_labels = result_df['Cluster'].tolist()
        self.assertTrue(
            all(label in [0, 1] for label in cluster_labels))  # Check if cluster labels are within expected range

    def test_apply_umap_custom_parameters(self):
        u_custom = apply_umap(self.df, n_neighbors=5, min_dist=0.5, n_components=3)
        self.assertEqual(u_custom.shape,
                         (self.df.shape[0], 3))  # Check if custom parameters result in correct output shape


if __name__ == '__main__':
    unittest.main()