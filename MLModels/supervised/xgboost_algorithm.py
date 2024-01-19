import json
import numpy as np
import pandas as pd
import xgboost as xgb
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import train_test_split

from MLModels.supervised.steel_model_interface import SteelModelInterface


def make_data_numeric(data):
    for column in data.columns:
        for index, value in data[column].items():
            if isinstance(value, str):
                data.at[index, column] = value.replace(",", ".")
    data = data.map(pd.to_numeric)
    return data


def load_steel_type_dict_from_json(path_encoder):
    with open(path_encoder, "r") as json_file:
        steel_types_dict = json.load(json_file)
    return steel_types_dict


class XGBoost(SteelModelInterface):
    def __init__(self):
        super().__init__()
        self.model = None
        self.y_test = None
        self.y_train = None
        self.X_test = None
        self.X_train = None
        self.y = None
        self.X = None
        self.data = None
        self.label_encoder = None

    def extract_steel_types(self):
        steels = {}
        counter = 0
        for i in range(len(self.data.iloc[:, -1])):
            if self.data.iloc[:, -1][i] not in steels:
                steels[self.data.iloc[:, -1][i]] = counter
                counter += 1
        return steels

    def convert_data(self):

        # Jeder Stahlklasse eine Nummer zuweisen
        self.label_encoder = self.extract_steel_types()
        self.data.iloc[:, -1] = [self.label_encoder[c] for c in self.data.iloc[:, -1]]

        # Jede Zelle numerisch machen
        for column in self.data.columns:
            for index, value in self.data[column].items():
                if isinstance(value, str):
                    self.data.at[index, column] = value.replace(",", ".")
        self.data = self.data.map(pd.to_numeric)

        # Aufteilung in data (X) und result (y)
        self.X = self.data.drop(self.data.columns[-1], axis=1)
        self.y = self.data.iloc[:, -1]

        # Aufteilung in Test- und Trainingsdaten
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2,
                                                                                random_state=42)

    def predict(self, instances):
        instances = make_data_numeric(instances)
        conv_instances = xgb.DMatrix(instances)
        try:
            results = self.model.predict(instances)
        except:
            results = self.model.predict(conv_instances)

        results = np.array([key for value in results for key, v in self.label_encoder.items() if v == value])
        return results

    def create_model(self, data):
        self.data = data
        self.convert_data()
        num_class = len(self.label_encoder.keys())

        # Parameter des Classifiers ermittelt durch Hyperparameter Optimization
        # siehe JupyterNotebook XGBoost_Supervised learning
        self.model = xgb.XGBClassifier(objective="multi:softmax",
                                       num_class=num_class,
                                       n_estimators=100,
                                       max_depth=3,
                                       learning_rate=1,
                                       gamma=0.1,
                                       reg_lambda=20)

        # Trainiere model
        self.model.fit(self.X_train, self.y_train)

    def test_model(self):
        # Evaluation mit Hilfe von 9-fold-Cross-Validation
        score_array = cross_val_score(
            self.model,
            self.X,
            self.y,
            cv=9)
        return "Accuracy XGBoost: " + str(sum(score_array) / len(score_array))

    def save_model(self, model_name, le_name):
        self.extract_steel_types()
        with open(le_name, "w") as json_file:
            json.dump(self.label_encoder, json_file)
        self.model.save_model(model_name)

    def load_model(self, path_model, path_encoder):
        self.data = path_model
        self.label_encoder = load_steel_type_dict_from_json(path_encoder)
        xgb2 = xgb.Booster()
        xgb2.load_model(self.data)
        self.model = xgb2
