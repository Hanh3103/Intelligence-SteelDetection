#!/usr/local/bin/python3
import importlib
import numpy as np


class SteelClassifier:

    def __init__(self, model):
        self.model = model

    def __init__(self, data_file_path, encoder_file_path, module_name, class_name, existing_model):
        self.data_file_path = data_file_path

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        self.model = model_class()
        if existing_model:
            self.model.load_model(data_file_path, encoder_file_path)
        else:
            self.model.create_model(data_file_path)

    def predict(self, instances):
        return self.model.predict(instances)

    def save_model(self, model_name, le_name):
        self.model.save_model(model_name, le_name)

    def retrain_model(self, file_name):
        self.model.retrain(file_name)

    def test_model(self):
        self.model.test()


my_classifier = SteelClassifier("test.h5", 'le.pkl', "supervised_test", "dlst", True)
test_data = [0.206166666666667, 0.241833333333333, 0.568666666666667, 0.0169166666666667, 0.0336666666666667,
             0.00798333333333333, 0.0506333333333333, 0.00323333333333333, 0.202666666666667
             ]
test_data = np.array(test_data).reshape(1, -1)
print(my_classifier.predict(test_data))
# my_classifier.save_model('test.h5', 'le.pkl')
