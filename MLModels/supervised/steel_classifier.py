import importlib


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
