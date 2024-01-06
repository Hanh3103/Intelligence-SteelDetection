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

from MLModels.supervised.steel_model_interface import SteelModelInterface

class dlst(SteelModelInterface):

    # CSV-Datei einlesen
    def __init__(self):
        super().__init__()


    def convert_data(self):
        # Aufteilen in Features (X) und Labels (y)
        X = self.data.iloc[:, :9].astype(float).values
        y = self.data.iloc[:, 9].values

        #Stahlklassen in numerische Werte
        self.label_encoder = LabelEncoder()
        y = self.label_encoder.fit_transform(y)
        label_mapping = dict(zip(self.label_encoder.classes_, range(len(self.label_encoder.classes_))))
        #print(label_mapping)

        #Aufteilen in Trainings- und Testdaten
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(X, y, test_size=0.2, random_state=42)


    def create_model(self, file_path):
        self.data = pd.read_csv(file_path)
        self.data.replace(',', '.', regex=True, inplace=True)
        self.convert_data()
        #Modell erstellen
        self.model = Sequential()
        self.model.add(Dense(64, activation='relu', input_dim=9))
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))

        #Modell kompilieren
        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])

        #Modell trainieren
        self.model.fit(self.X_train, self.y_train, epochs=100, batch_size=32, validation_data=(self.X_test, self.y_test))



    def predict(self, instances):
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

    def retrain_model(self, newdata_file):
        #self.model.pop()
        #self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))
        #for layer in self.model.layers[:-1]:
        #    layer.trainable = False
        #self.model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])
        #self.model.fit(new_data, new_labels, epochs=100, batch_size=32)
        pass



    def test(self):
        # Modell evaluieren
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        print(f'Loss: {loss}, Accuracy: {accuracy}')

        #Vorhersagen
        #print(type(X_test[0]))
        predictions = self.model.predict(self.X_test)
        predicted_classes = predictions.argmax(axis=-1)
        print(self.label_encoder.inverse_transform(self.y_test))
        print(self.label_encoder.inverse_transform(predicted_classes))

        #einzelner Test
        test_data = [0.206166666666667,	0.241833333333333,	0.568666666666667,	0.0169166666666667,	0.0336666666666667,	0.00798333333333333,	0.0506333333333333,	0.00323333333333333,	0.202666666666667
        ] # 19 St 48
        array = np.array(test_data).reshape(1, -1)
        pred = self.model.predict(array)
        print(pred)
        print(self.label_encoder.inverse_transform(pred.argmax(axis=-1)))
