import importlib
import numpy as np
import pandas as pd


class SteelClassifier:

    def __init__(self, data_frame, encoder_file_path, module_name, class_name, existing_model):
        """
        Initializes the SteelClassifier with data, imports and initializes the model, and optionally loads a pre-trained model.

        Args:
            data_frame: DataFrame, input data for model initialization.
            encoder_file_path: str, path to the label encoder file.
            module_name: str, name of the module containing the model class.
            class_name: str, name of the model class.
            existing_model: bool, indicating whether an existing model should be loaded.
        """
        module = importlib.import_module(module_name)
        model_class = getattr(module, class_name)
        self.model = model_class()

        if existing_model:
            self.model.load_model(data_frame, encoder_file_path)
        else:
            self.model.create_model(data_frame)

    def predict(self, instances):
        """
        Predicts labels and probabilities for given instances.

        Args:
            instances: DataFrame or numpy array, instances to predict.

        Returns:
            prediction_label: array, predicted labels.
            probabilities: array, predicted probabilities for each class.
        """
        return self.model.predict(instances)

    def save_model(self, model_name, le_name):
        """
        Saves the trained model and label encoder to files.

        Args:
            model_name: str, filename for saving the model.
            le_name: str, filename for saving the label encoder.
        """
        self.model.save_model(model_name, le_name)

    def test_model(self):
        """
        Calls the evaluation of the performance of the trained model.

        Returns:
            str: Evaluation results.
        """
        return self.model.test_model()

