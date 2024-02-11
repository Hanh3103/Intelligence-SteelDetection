from enum import Enum


class AlgorithmSupervised(Enum):
    """
    Enum class for Supervised Learning Algorithms.

    Each enum member represents a supervised learning algorithm with associated properties.

    Attributes:
        - algorithm (str): The identifier or name of the algorithm.
        - model_format (str): The format of the model file.
        - encoder_format (str): The format of the encoder file.
        - class_name (str): The class name associated with the algorithm.

    Enum Members:
        - DeepLearning: Represents the Deep Learning algorithm.
        - Xgboost: Represents the XGBoost algorithm.
    """

    # Deep Learning algorithm
    DeepLearning = {
        "algorithm": "deep_learning",
        "model_format": ".h5",
        "encoder_format": ".pkl",
        "class_name": "DeepLearning1"
    }  # only use format .h5 and .pkl

    # XGBoost algorithm
    Xgboost = {
        "algorithm": "xgboost_algorithm",
        "model_format": ".json",
        "encoder_format": ".json",
        "class_name": "XGBoost"
    }  # only use format .json and .json
