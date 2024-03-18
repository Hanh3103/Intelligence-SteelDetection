import joblib
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras import regularizers
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Dropout
from tensorflow.keras.layers import Sequential
from tensorflow.keras.layers import load_model
from sklearn.metrics import f1_score

from MLModels.supervised.steel_model_interface import SteelModelInterface


class DeepLearning1(SteelModelInterface):

    def __init__(self):
        super().__init__()

    def convert_data(self):
        """
        Converts the input data into features (X) and labels (y), encoding labels into numerical values,
        and splits the data into training and testing sets.
        """
        self.X = self.data.iloc[:, :-1].astype(float).values
        self.y = self.data.iloc[:, -1].values

        self.label_encoder = LabelEncoder()
        self.y = self.label_encoder.fit_transform(self.y)

        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

    def create_model(self, data):
        """
        Creates the deep learning model, compiles it, and trains it on the provided data.

        Args:
            data: DataFrame, input data for model training and evaluation.
        """
        self.model_fresh = True

        self.data = data
        self.convert_data()
        self.model = Sequential()
        self.model.add(Dense(248, activation='relu', input_dim=self.data.shape[1] - 1))
        self.model.add(Dense(124, activation='relu'))
        self.model.add(Dropout(0.5))

        self.model.add(Dense(64, activation='relu', kernel_regularizer=regularizers.l2(0.05)))

        self.model.add(Dense(32, activation='relu'))

        self.model.add(Dense(len(self.label_encoder.classes_), activation='softmax'))

        self.model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])


        self.model.fit(self.X_train, self.y_train, epochs=800, batch_size=8, validation_data=(self.X_test, self.y_test))

    def predict(self, instances):
        """
        Predicts labels and probabilities for given instances.

        Args:
            instances: DataFrame or numpy array, instances to predict.

        Returns:
            prediction_label: array, predicted labels.
            probabilities: array, predicted probabilities for each class.
        """

        try:
            instances = instances.to_numpy(dtype=np.float64)
        except:
            pass
        predictions = self.model.predict(instances)
        predicted_classes = predictions.argmax(axis=-1)
        prediction_label = self.label_encoder.inverse_transform(predicted_classes)
        probabilities = predictions.max(axis=-1)
        return prediction_label, probabilities

    def save_model(self, model_name, le_name):
        """
        Saves the trained model and label encoder to files.

        Args:
             model_name: str, filename for saving the model.
            le_name: str, filename for saving the label encoder.
        """
        self.model.save(model_name)
        joblib.dump(self.label_encoder, le_name)

    def load_model(self, path_model, path_encoder):
        """
        Loads a trained model and label encoder from files.

        Args:
            path_model: str, path to the saved model file.
            path_encoder: str, path to the saved label encoder file.
        """
        self.model_fresh = False
        self.model = load_model(path_model)
        self.label_encoder = joblib.load(path_encoder)

    def test_model(self):
        """
        Evaluates the performance of the trained model on test and all data.

        Returns:
            str: Evaluation results including loss, accuracy, and F1-score.
        """
        if not self.model_fresh:
            return f'Evaluation not available for loaded models'
        loss, accuracy = self.model.evaluate(self.X_test, self.y_test)
        all_loss, all_accuracy = self.model.evaluate(self.X, self.y)
        all_f1 = f1_score(self.y, self.label_encoder.fit_transform(self.predict(self.X)[0]), average='macro')
        test_f1 = f1_score(self.y_test, self.label_encoder.fit_transform(self.predict(self.X_test)[0]), average='macro')
        return f'Evaluation test data:\nLoss: {loss}, Accuracy: {accuracy}, F1-Score: {test_f1}\nEvaluation all data:\nLoss: {all_loss}, Accuracy: {all_accuracy}, F1-Score: {all_f1}'
