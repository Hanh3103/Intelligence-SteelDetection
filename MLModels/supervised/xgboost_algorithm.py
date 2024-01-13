import pandas as pd
import xgboost as xgb
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
from sklearn.model_selection import cross_val_score
from MLModels.supervised.steel_model_interface import SteelModelInterface


class XGBoost(SteelModelInterface):
    def __init__(self):
        super().__init__()

    def make_data_numeric(self, data):
        for column in data.columns:
            for index, value in data[column].items():
                if isinstance(value, str):
                    data.at[index, column] = value.replace(",", ".")
        data = data.map(pd.to_numeric)
        return data

    def convert_data(self):

        # Jeder Stahlklasse eine Nummer zuweisen
        classes_dict = {"St 48": 0, "St 37": 1, "St Si": 2, "Flussstahl": 3, "Puddelstahl": 4}
        self.data["steel_class"] = [classes_dict[c] for c in self.data["steel_class"]]

        # Jede Zelle numerisch machen
        for column in self.data.columns:
            for index, value in self.data[column].items():
                if isinstance(value, str):
                    self.data.at[index, column] = value.replace(",", ".")
        self.data = self.data.map(pd.to_numeric)

        # Aufteilung in data (X) und result (y)
        self.X = self.data.drop(columns="steel_class")
        self.y = self.data["steel_class"]

        # Aufteilung in Test- und Trainingsdaten
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def predict(self, instances):
        instances = self.make_data_numeric(instances)
        if type(self.data) != "<class 'pandas.core.frame.DataFrame'>":
            instances = xgb.DMatrix(instances)
        return self.model.predict(instances)

    def create_model(self, data):
        self.data = data
        self.convert_data()

        # Parameter des Classifiers ermittlet durch Hyperparameter Optimization
        # siehe JupyterNotebook XGBoost_Supervisedlearning
        self.model = xgb.XGBClassifier(objective="multi:softmax",
                                       num_class=5,
                                       max_depth=3,
                                       learning_rate=0.05,
                                       gamma=0.01)

        # Trainiere model
        self.model.fit(self.X_train, self.y_train)

    def test_model(self):
        # Evaluation mit Hilfe von 9-fold-Cross-Validation

        score_array = cross_val_score(
            xgb.XGBClassifier(objective="multi:softmax",
                              num_class=5,
                              max_depth=3,
                              learning_rate=0.05,
                              gamma=0.01),
            self.X,
            self.y,
            cv=9)
        return "Accuracy XGBoost: " + str(sum(score_array) / len(score_array))

    def save_model(self, model_name, le_name):
        self.model.save_model(model_name)

    def load_model(self, path_model, path_encoder):
        self.data = path_model
        xgb2 = xgb.Booster()
        xgb2.load_model(self.data)
        self.model = xgb2
        print(self.model.feature_names)


