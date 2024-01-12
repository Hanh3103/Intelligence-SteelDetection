#!/usr/local/bin/python3
import importlib
import numpy as np
import pandas as pd
class SteelClassifier:


    def __init__(self, model):
        self.model = model


    def __init__(self, data_frame, encoder_file_path, module_name, class_name, existing_model):

        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        self.model = model_class()
        if existing_model:
            self.model.load_model(data_frame, encoder_file_path)
        else:
            self.model.create_model(data_frame)



    def predict(self, instances):
        return self.model.predict(instances)


    def save_model(self, model_name, le_name):
        self.model.save_model(model_name, le_name)

    def test_model(self):
        return self.model.test_model()



#data = pd.read_csv('labeled_OES_data.csv')
#data.replace(',', '.', regex=True, inplace=True)
#
#my_classifier = SteelClassifier(data, 'le.pkl', "deep_learning", "DeepLearning1", False)
#test_data = [0.0609833333333333,0.005,0.312333333333333,0.0199166666666667,0.03495,0.0249166666666667,0.0465666666666667,0.002,0.146833333333333
#
#             ]
##test_data = np.array(test_data).reshape(1, -1)
##print(my_classifier.predict(test_data))
#print(my_classifier.test_model())
##my_classifier.save_model('test.h5', 'le.pkl')


'''
data = pd.read_csv('labeled_OES_data.csv')
data.replace(',', '.', regex=True, inplace=True)
#data = data.drop(columns="steel_class")
my_classifier = SteelClassifier(data, '', "xgboost_algorithm", "XGBoost", False)


test_data = {
    "C": [0.0609833333333333],
    "Si": [0.005],
    "Mn": [0.312333333333333],
    "P": [0.0199166666666667],
    "S": [0.03495],
    "Cr": [0.0249166666666667],
    "Ni": [0.0465666666666667],
    "Al": [0.002],
    "Cu": [0.146833333333333]
}
df = pd.DataFrame(test_data)
#print(my_classifier.predict(data))
print(my_classifier.predict(df))
#my_classifier.save_model('xgboost_model.json', '')
'''