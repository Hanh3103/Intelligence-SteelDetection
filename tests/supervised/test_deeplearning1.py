import unittest

import numpy as np
import pandas as pd

from MLModels.supervised.steel_classifier import SteelClassifier


class TestDeeplearning1(unittest.TestCase):

    def test_model_test(self):
        data = pd.read_csv('labeled_OES_data.csv')
        data.replace(',', '.', regex=True, inplace=True)
        test_model = SteelClassifier(data, '', "MLModels.supervised.deep_learning", "DeepLearning1", False)
        print(test_model.test_model())
        self.assertIsInstance(test_model.test_model(), str)

    def test_predict(self):
        data = pd.read_csv('labeled_OES_data.csv')
        data.replace(',', '.', regex=True, inplace=True)
        test_model = SteelClassifier(data, '', "MLModels.supervised.deep_learning", "DeepLearning1", False)
        test_data = [0.0609833333333333, 0.005, 0.312333333333333, 0.0199166666666667, 0.03495, 0.0249166666666667, 0.0465666666666667, 0.002, 0.146833333333333]
        test_data = np.array(test_data).reshape(1, -1)
        self.assertEqual(test_model.predict(test_data)[0][0], 'St 37')
        self.assertGreater(test_model.predict(test_data)[1][0], 0.8)

    def test_save_model(self):
        data = pd.read_csv('labeled_OES_data.csv')
        data.replace(',', '.', regex=True, inplace=True)
        test_model = SteelClassifier(data, '', "MLModels.supervised.deep_learning", "DeepLearning1", False)
        self.assertIsNone(test_model.save_model("test_model.h5", "test_label_encoder.pkl"))

    def test_load_model(self):
        self.assertIsInstance(SteelClassifier('test_model.h5', 'test_label_encoder.pkl', "MLModels.supervised.deep_learning", "DeepLearning1", True), SteelClassifier)



if __name__ == '__main__':
    unittest.main()