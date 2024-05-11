import unittest
import pandas as pd
from MLModels.data_augmentation.data_augmentation_ensemble import Data_Augmentation_Ensemble

class TestDataAugmentation(unittest.TestCase):

    def test_wgan(self):
        original_data = pd.read_csv("labeled_OES_data_2.csv")
        data_augmentation = Data_Augmentation_Ensemble().augment_steel_data("labeled_OES_data_2.csv")
        self.assertTrue(len(original_data < data_augmentation))