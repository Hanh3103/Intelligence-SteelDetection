import unittest
import pandas as pd
import numpy as np
from sklearn.mixture import GaussianMixture
from MLModels.unsupervised.cluster_Algorithm import gmm_algorithm, recommend_gmm_components

class TestSteelClassification(unittest.TestCase):

    def setUp(self):
        # Read data from CSV for testing
        csv_file_path = 'test_data_unsupervised.csv'
        self.df = pd.read_csv(csv_file_path)

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
        numeric_columns = self.df.select_dtypes(include=[np.number]).columns
        recommended_components = recommend_gmm_components(self.df[numeric_columns])

        # Test if the recommended number is within a reasonable range
        self.assertGreaterEqual(recommended_components, 1)
        self.assertLessEqual(recommended_components, 10)



    def test_csv_data_types(self):
        # Test if the data types in the CSV file are as expected
        expected_data_types = {'Fe': np.float64, 'C': np.float64, 'Si': np.float64, 'Mn': np.float64,
                                   'P': np.float64, 'S': np.float64,
                                   'Cr': np.float64, 'Mo': np.float64, 'Ni': np.float64, 'Al': np.float64,
                                   'Co': np.float64, 'Cu': np.float64,
                                   'Nb': np.float64, 'Ti': np.float64, 'V': np.float64, 'W': np.float64,
                                   'Pb': np.float64, 'Sn': np.float64,
                                   'B': np.float64, 'Ca': np.float64, 'Zr': np.float64, 'As': np.float64,
                                   'Bi': np.float64}
        self.assertTrue(all(self.df[column].dtype == expected_data_types[column] for column in expected_data_types))

    def test_recommend_gmm_components_valid_range(self):
         # Test if the recommended number of components is within a reasonable range
         numeric_columns = self.df.select_dtypes(include=[np.number]).columns
         recommended_components = recommend_gmm_components(self.df[numeric_columns])
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









