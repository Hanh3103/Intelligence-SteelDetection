import unittest
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from MLModels.unsupervised.cluster_Algorithm import gmm_algorithm, recommend_gmm_components


class TestSteelClassification(unittest.TestCase):

    def setUp(self):

        # create sample data
        data = {
            'Grade': ['St 48'] * 21,
            'ID': ['20_21_1', '20_23_1', '20_28_1', '20_31_1', '20_32_1', '20_41_1', '21_11_1', '21_14_1', '21_31_1',
                   '21_32_1', '21_41_1', '24_11_1', '24_12_1', '24_31_1', '24_32_1', '24_41_1', '25_11_1', '25_14_1',
                   '25_31_1', '25_32_1', '25_41_1'],
            'Fe': [0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0,
                   0.0],
            'C': [0.0331, 0.162, 0.144, 0.264, 0.255, 0.248, 0.21, 0.169, 0.154, 0.23, 0.213, 0.194, 0.202, 0.184,
                  0.222, 0.302, 0.206, 0.201, 0.276, 0.186, 0.27],
            'S': [0.204, 0.13, 0.265, 0.274, 0.288, 0.197, 0.241, 0.227, 0.28, 0.209, 0.005, 0.231, 0.328, 0.128, 0.271,
                  0.204, 0.422, 0.137, 0.296, 0.268, 0.2],
        }
        self.data = pd.DataFrame(data)
        self.n_components = 3



    def test_recommend_gmm_components(self):
        # Test if the recommended number of components is an integer
        # Exclude non-numeric columns from the selection
        numeric_columns = self.data.select_dtypes(include=[np.number]).columns
        recommended_components = recommend_gmm_components(self.data[numeric_columns])

        # Test if the recommended number is within a reasonable range
        self.assertGreaterEqual(recommended_components, 1)
        self.assertLessEqual(recommended_components, 10)



    def test_recommend_gmm_components_valid_range(self):
         # Test if the recommended number of components is within a reasonable range
         numeric_columns = self.data.select_dtypes(include=[np.number]).columns
         recommended_components = recommend_gmm_components(self.data[numeric_columns])
         self.assertGreaterEqual(recommended_components, 1)
         self.assertLessEqual(recommended_components, 10)

    def test_gmm_algorithm_output_type(self):
        # Check the type of the return value
        result = gmm_algorithm(self.data, self.n_components)
        self.assertIsInstance(result, pd.DataFrame)

    def test_gmm_algorithm_output_columns(self):
        # Check if the output contains the 'Cluster' column
        result = gmm_algorithm(self.data, self.n_components)
        self.assertIn('Cluster', result.columns)

    def test_gmm_algorithm_number_of_clusters(self):
        # Check if the number of returned clusters is correct
        result = gmm_algorithm(self.data, self.n_components)
        unique_clusters = result['Cluster'].unique()
        self.assertEqual(len(unique_clusters), self.n_components)

    def test_gmm_algorithm_fit_predict(self):
        # Check if the GMM model is applied properly
        result = gmm_algorithm(self.data, self.n_components)
        gmm = GaussianMixture(n_components=self.n_components, random_state=0)
        expected_clusters = gmm.fit_predict(self.data[['Fe', 'C', 'S']])  # Adjust the selected columns according to your requirements
        self.assertTrue((result['Cluster'] == expected_clusters).all())

    def test_gmm_algorithm_output_shape(self):
            """
            Test if the output shape is correct.
            """
            result = gmm_algorithm(self.data, self.n_components)
            self.assertEqual(result.shape[0], self.data.shape[0])

    def test_gmm_algorithm_cluster_values(self):
            """
            Test if the values in the 'Cluster' column are valid cluster IDs.
            """
            result = gmm_algorithm(self.data, self.n_components)
            unique_clusters = result['Cluster'].unique()
            expected_unique_clusters = range(self.n_components)
            self.assertSetEqual(set(unique_clusters), set(expected_unique_clusters))


if __name__ == '__main__':
    unittest.main()









