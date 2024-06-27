import pandas as pd
from MLModels.data_augmentation.WGAN import WGAN_steel_type_data_augmentation
from MLModels.data_augmentation.SMOTE_ENN import Steel_Smote_Enn


class Data_Augmentation_Ensemble:

    """
       Class to perform data augmentation on steel type data using SMOTE-ENN and WGAN methods.
    """
    def resample_steel_data_smote_enn(self, data_path):
        """
                Quickly augment steel data using SMOTE-ENN and save the augmented data to a new CSV file.

                Args:
                    data_path (str): Path to the CSV file containing the original data.
        """
        data = pd.read_csv(data_path)
        data = self.prepare_data(data)
        data = Steel_Smote_Enn(data).apply()
        data.to_csv(data_path[:-4] + "_with_augmented_data_SMOTE_ENN.csv", index=False)

    def augment_steel_data_WGANs(self, data_path, min_amount_samples_per_class, max_amount_samples_per_class):
        """
                Realistically augment steel data using WGAN and save the augmented data to a new CSV file.

                Args:
                    data_path (str): Path to the CSV file containing the original data.
        """
        data = pd.read_csv(data_path)
        data = self.prepare_data(data)
        dfs_by_steeltype = self.group_by_steel_type(data)
        amount_to_augment_dict = self.calc_amounts(dfs_by_steeltype, min_amount_samples_per_class, max_amount_samples_per_class)
        for steel_type in dfs_by_steeltype.keys():
            if steel_type in amount_to_augment_dict.keys():
                augmented_data = self.augment_for_steel_type(amount=amount_to_augment_dict[steel_type], data=dfs_by_steeltype[steel_type], steel_type=steel_type)
                dfs_by_steeltype[steel_type] = pd.concat([dfs_by_steeltype[steel_type], pd.DataFrame(augmented_data, columns=dfs_by_steeltype[steel_type].columns)], ignore_index=True)
            print("Augmented: ", steel_type)
        self.save_new_augmented_dataset(dfs_by_steeltype, data_path, data.columns)


    """
            Realistically augment data for a single class using WGAN and save the augmented data to a new CSV file.
    """
    def single_class_WGAN_data_augmentation(self, data_path, steel_type, amount):
        data = pd.read_csv(data_path)
        data = self.prepare_data(data)
        dfs_by_steeltype = self.group_by_steel_type(data)
        augmented_data = self.augment_for_steel_type(amount, dfs_by_steeltype[steel_type])
        data_single_class = pd.DataFrame(augmented_data, columns=dfs_by_steeltype[steel_type].columns)
        data_single_class[data.columns[-1]] = steel_type
        data_single_class.to_csv("single_augmented_" + steel_type + "_samples", index=False)

    def group_by_steel_type(self, data):
        """
                Groups the data by steel type.

                Args:
                    data (DataFrame): The input data.

                Returns:
                    dict: A dictionary where keys are steel types and values are DataFrames of the grouped data.
        """
        element_data_grouped_by_steeltype = data.groupby(data.iloc[:, -1])
        dfs_by_steeltype = {}

        for type, group_data in element_data_grouped_by_steeltype:
            dfs_by_steeltype[type] = group_data.iloc[:, :-1]
        return dfs_by_steeltype

    def calc_amounts(self, dfs_by_steeltype, min_amount, max_amount):
        """
                Calculates the number of samples to augment for each steel type.

                Args:
                    dfs_by_steeltype (dict): Dictionary of grouped data by steel type.

                Returns:
                    dict: A dictionary where keys are steel types and values are the number of samples to augment.
        """
        amount_to_dict = {}
        for steel_type in dfs_by_steeltype.keys():
            if len(dfs_by_steeltype[steel_type]) < min_amount:
                amount_to_dict[steel_type] = min_amount - len(dfs_by_steeltype[steel_type])
            elif len(dfs_by_steeltype[steel_type]) > max_amount:
                over_limit = len(dfs_by_steeltype[steel_type]) - max_amount
                dfs_by_steeltype[steel_type] = dfs_by_steeltype[steel_type].drop(dfs_by_steeltype[steel_type].index[:over_limit])

        return amount_to_dict

    def augment_for_steel_type(self, amount, data, steel_type):
        """
                Augments data for a specific steel type using WGAN.

                Args:
                    amount (int): Number of samples to augment.
                    data (DataFrame): Data of the specific steel type.
                    steel_type (str): Type of steel.

                Returns:
                    ndarray: Augmented data.
        """
        wgan_steel = WGAN_steel_type_data_augmentation(len(data.columns))
        return wgan_steel.augment_data(amount, data, steel_type)

    def save_new_augmented_dataset(self, dfs_by_steeltype, data_path, column_labels):
        """
                Saves the augmented dataset to a new CSV file.

                Args:
                    dfs_by_steeltype (dict): Dictionary of augmented data grouped by steel type.
                    data_path (str): Path to save the new CSV file.
                    column_labels (Index): Column labels of the original data.
        """
        augmented_dataset = pd.DataFrame()
        for steel_type, data_frame in dfs_by_steeltype.items():
            steel_col = [steel_type for _ in range(len(data_frame))]
            data_frame[column_labels[-1]] = steel_col
            augmented_dataset = pd.concat([augmented_dataset, data_frame], ignore_index=True)
        augmented_dataset.columns = column_labels
        augmented_dataset.to_csv(data_path + "_with_augmented_data_WGANs.csv", index=False)

    def prepare_data(self, data):
        """
                Prepares the data by converting all values to numeric.

                Args:
                    data (DataFrame): The input data.

                Returns:
                    DataFrame: The prepared data with numeric values.
        """
        # make data numeric
        for column in data.columns:
            for index, value in data[column].items():
                if isinstance(value, str):
                    data.at[index, column] = value.replace(",", ".")
        return data