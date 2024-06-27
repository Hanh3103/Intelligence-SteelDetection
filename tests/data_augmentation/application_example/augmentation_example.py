from MLModels.data_augmentation.data_augmentation_techniques import Data_Augmentation_Ensemble

data_augmentation = Data_Augmentation_Ensemble()
# fast augmentation SMOTE-ENN
smote_enn_dataset = data_augmentation.resample_steel_data_smote_enn(data_path="../example_dataset.csv")

# realistic augmentation WGAN
wgan_dataset = data_augmentation.augment_steel_data_WGANs(data_path="../example_dataset.csv")
