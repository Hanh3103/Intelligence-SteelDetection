from enum import Enum


class AlgorithmSupervised(Enum):
    # Algorithms for Supervised Learning
    DeepLearning = {"algorithm": "deep_learning",
                    "model_format": ".h5",
                    "encoder_format": ".pkl",
                    "class_name": "DeepLearning1"}  # only use format .h5 and .pkl
    Xgboost = {"algorithm": "xgboost_algorithm",
               "model_format": ".json",
               "encoder_format": ".json",
               "class_name": "XGBoost"}  # only use format .json and .json
