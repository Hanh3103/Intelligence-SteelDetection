### Jonathan Püschel
# Bachelor Thesis: Data Augmentation for Steel Classification

This project explores the implementation of various data augmentation techniques to enhance the classification accuracy of steel types in imbalanced datasets. The primary methods investigated include:

1. **Generative Adversarial Networks (GANs):** While traditional GANs faced challenges in capturing the diversity of steel classes, resulting in low-quality data, the improved Wasserstein GANs (WGANs) successfully generated realistic data samples that enhanced the training process.

2. **Wasserstein Generative Adversarial Networks (WGANs):** WGANs addressed the limitations of traditional GANs by effectively replicating the underlying data distribution. This technique significantly improved the quality of the generated samples, contributing to better model performance.

3. **Synthetic Minority Over-sampling Technique with Edited Nearest Neighbors (SMOTE-ENN):** SMOTE-ENN effectively balanced the dataset by oversampling minority classes and removing noisy data, which improved the classification performance of the supervised learning models.



### Key Findings
- Steel datasets can be imbalanced, which can lead to poor performance of supervised learning classifications overall or on singular classes.
- To fix bad performance on singular classes we implemented WGANs to augment data for only that class. The influence on supervised learning needs to be examined to ensure desired results. 
- For overall bad performance we can either use WGANs for each steel class or SMOTE-ENN to enhance the classification abilities of supervised learning models.


### Implementation in this project

The implementation can be found in the following directories:

**MLModels -> data_augmentation:** In this directory we implemented the main functionalities for Data Augmentation. The file data_augmentation_techniques.py supplies the methods that execute SMOTE-ENN and augment data with WGANs. The following three functions can be applied to the dataset.
1. resample_steel_data_smote_enn(self, data_path): Quickly augments steel data using the SMOTE-ENN technique. Reads the original data from a CSV file, prepares it, applies SMOTE-ENN to balance the dataset, and saves the augmented data to a new CSV file. Args: data_path (str): Path to the CSV file containing the original data.


2. augment_steel_data_WGANs(self, data_path, min_amount_samples_per_class, max_amount_samples_per_class):
Realistically augments steel data using Wasserstein Generative Adversarial Networks (WGANs).
Reads the original data from a CSV file, prepares it, calculates the number of samples needed for augmentation for each steel type, applies WGANs to generate realistic samples, and saves the augmented data to a new CSV file. Args:
data_path (str): Path to the CSV file containing the original data.
min_amount_samples_per_class (int): Minimum number of samples per class for augmentation.
max_amount_samples_per_class (int): Maximum number of samples per class for augmentation.

3. single_class_WGAN_data_augmentation(self, data_path, steel_type, amount):
Realistically augments data for a single steel type class using WGAN.
Reads the original data from a CSV file, prepares it, applies WGAN to generate the specified amount of samples for the given steel type, and saves the augmented data to a new CSV file. Args: data_path (str): Path to the CSV file containing the original data.
steel_type (str): The steel type class to be augmented.
amount (int): The number of samples to generate for the specified steel type.

**tests-> data_augmentation**
In this directory we tested the three presented methods to ensure the functionality of the program. As a testing set we use the provided dataset and check if the composition changes for each data augmentation application.


**BA_Jonathan_Pueschel**
Within this directory there are protocols, visualizations and examples, which the thesis is based on, together with a PDF-file of the thesis. The directory comprises the following subdirectories:
1. **FiguresBA:** includes all figures depicted in the thesis
2. **GAN_Training_In_Pycharm:** Given the time-intensive nature of training GANs in Jupyter Notebooks, we opted to train the example GANs for the thesis in PyCharm. Note that this directory does not contain significant code used in the thesis, as all the essential code is located in the "Protocol JupyterNotebooks" directory.
3. **Protocol JupyterNotebooks:** For each data augmentation technique we created a Jupyter Notebook to protocol the training and functionalities
4. **Supervised Evaluation:** This directory contains the evaluation of the supervised learning models before and after the use of the presented data augmentation techniques from the thesis.

