#!/usr/local/bin/python3
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from sklearn.preprocessing import LabelEncoder
import numpy as np
from tensorflow.keras.models import load_model
import joblib
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dropout
from tensorflow.keras import regularizers
from MLModels.supervised.steel_model_interface import SteelModelInterface


class DeepLearning1(SteelModelInterface):

    def __init__(self):
        super().__init__()

    def convert_data(self):
        # Aufteilen in Features (X) und Labels (y)
        self.X = self.data.iloc[:, :-1].astype(float).values
        self.y = self.data.iloc[:, -1].values

        # Stahlklassen in numerische Werte
        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)
        # label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        # print(label_mapping)

        # Aufteilen in Trainings- und Testdaten
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def create_model(self, data):
        self.data = data
        #self.data = pd.read_csv(file_path)
        #self.data.replace(',', '.', regex=True, inplace=True)
        self.convert_data()
        # Modell erstellen
        self.model = Sequential()
        self.model.add(Dense(248, activation='relu', input_dim=self.data.shape[1] - 1))
        self.model.add(Dense(124, activation='relu'))
        self.model.add(Dropout(0.5))

        #L2-Regularisierung
        self.model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.01)))

        self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))

        # Modell kompilieren
        #optimizer = adam_v2(learning_rate=0.001)
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        # Modell trainieren

        self.model.fit(self.X_train, self.y_train, epochs=400, batch_size=8,
                       validation_data=(self.X_test, self.y_test))

    def predict(self, instances):
        instances = np.array(instances).reshape(1, -1)
        predictions = self.model.predict(instances)
        predicted_classes = predictions.argmax(axis=-1)
        prediction_label = self.label_encoder.inverse_transform(predicted_classes)
        probabilities = predictions.max(axis=-1)
        return prediction_label, probabilities

    def save_model(self, model_name, le_name):
        self.model.save(model_name)
        joblib.dump(self.label_encoder, le_name)

    def load_model(self, path_model, path_encoder):
        self.model = load_model(path_model)
        self.label_encoder = joblib.load(path_encoder)

    def test_model(self):
        # Modell evaluieren
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        all_loss, all_accuracy = self.model.evaluate(self.X, self.y)
        #print(f'Loss: {loss}, Accuracy: {accuracy}')
        return f'Evaluation test data:\nLoss: {loss}, Accuracy: {accuracy}\nEvaluation all data:\nLoss: {all_loss}, Accuracy: {all_accuracy}'

        # Vorhersagen
        # print(type(X_test[0]))
        #predictions = self.model.predict(self.X_test)
        #predicted_classes = predictions.argmax(axis=-1)
        #print(self.label_encoder.inverse_transform(self.y_test))
        #print(self.label_encoder.inverse_transform(predicted_classes))
#
        ## einzelner Test
        #test_data = [0.206166666666667, 0.241833333333333, 0.568666666666667, 0.0169166666666667, 0.0336666666666667,
        #             0.00798333333333333, 0.0506333333333333, 0.00323333333333333, 0.202666666666667
        #             ]  # 19 St 48
        #array = np.array(test_data).reshape(1, -1)
        #pred = self.model.predict(array)
        #print(pred)
        #print(self.label_encoder.inverse_transform(pred.argmax(axis=-1)))
