
class SteelModelInterface:

    def __init__(self):
        pass

    def predict(self, instances):
        pass

    def create_model(self, file_path):
        pass

    def test_model(self):
        pass

    def save_model(self, model_name, le_name):
        pass

    def load_model(self, path_model, path_encoder):
        pass
