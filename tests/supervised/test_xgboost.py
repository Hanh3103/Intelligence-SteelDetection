import unittest
import pandas as pd

from MLModels.supervised.steel_classifier import SteelClassifier


class Test_xgboost(unittest.TestCase):

    def test_model_test(self):
        data = pd.read_csv('labeled_OES_data.csv')
        test_model = SteelClassifier(data, '', "MLModels.supervised.xgboost_algorithm", "XGBoost", False)
        print(test_model.test_model())
        self.assertIsInstance(test_model.test_model(), str)

    def test_predict(self):
        data = pd.read_csv('labeled_OES_data.csv')
        test_model = SteelClassifier(data, '', "MLModels.supervised.xgboost_algorithm", "XGBoost", False)
        test_data = [[0.0609833333333333, 0.005, 0.312333333333333, 0.0199166666666667, 0.03495, 0.0249166666666667, 0.0465666666666667, 0.002, 0.146833333333333]]
        self.assertEqual(test_model.predict(test_data), 'St 37')

    def test_save_model(self):
        data = pd.read_csv('labeled_OES_data.csv')
        test_model = SteelClassifier(data, '', "MLModels.supervised.xgboost_algorithm", "XGBoost", False)
        self.assertIsNone(test_model.save_model("test_save_model_xgboost.json", "test_save_labelenc_xgboost.json"))

    def test_load_model(self):
        self.assertIsInstance(SteelClassifier('test_save_model_xgboost.json', 'test_save_labelenc_xgboost.json',
                                              "MLModels.supervised.xgboost_algorithm", "XGBoost", True),
                              SteelClassifier)


if __name__ == '__main__':
    unittest.main()
