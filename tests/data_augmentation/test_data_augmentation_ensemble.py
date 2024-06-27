import unittest
import pandas as pd
from MLModels.data_augmentation.data_augmentation_techniques import Data_Augmentation_Ensemble


class TestDataAugmentation(unittest.TestCase):
    """
       Unit tests for the Data_Augmentation_Ensemble class.
    """
    def test_wgan_overall(self):
        """
               Test the WGAN-based augmentation method to ensure the augmented data has more rows than the original data.
        """
        original_data = pd.read_csv("labeled_OES_data_2.csv")
        data_augmentation = Data_Augmentation_Ensemble().augment_steel_data_WGANs("labeled_OES_data_2.csv", 300, 500)
        self.assertTrue(len(original_data < data_augmentation))

    def test_wgan_singular_class(self):
        """
               Test the WGAN-based augmentation method to ensure the augmented data has more rows than the original data.
        """
        original_data = pd.read_csv("labeled_OES_data_2.csv")
        data_augmentation = Data_Augmentation_Ensemble().augment_steel_data_WGANs("labeled_OES_data_2.csv", 300, 500)
        self.assertTrue(len(original_data < data_augmentation))

    def test_SMOTE_ENN(self):
        """
                Test the SMOTE-ENN-based augmentation method to ensure the augmented data has more rows than the original data.
        """
        original_data = pd.read_csv("labeled_OES_data_2.csv")
        Data_Augmentation_Ensemble().resample_steel_data_smote_enn("labeled_OES_data_2.csv")
        data_augmentation = pd.read_csv("labeled_OES_data_2_with_augmented_data_SMOTE_ENN.csv")
        self.assertTrue(len(original_data) < len(data_augmentation))