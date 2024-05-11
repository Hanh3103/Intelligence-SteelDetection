import pandas as pd
from MLModels.data_augmentation.WGAN import WGAN_steel_type_data_augmentation


class Data_Augmentation_Ensemble:

    def augment_steel_data(self, data_path):
        data = pd.read_csv(data_path)
        data = self.prepare_data(data)
        dfs_by_steeltype = self.group_by_steel_type(data)
        amount_to_augment_dict = self.calc_amounts(dfs_by_steeltype)
        for steel_type in dfs_by_steeltype.keys():
            if steel_type in amount_to_augment_dict.keys():
                augmented_data = self.augment_for_steel_type(amount=amount_to_augment_dict[steel_type], data=dfs_by_steeltype[steel_type], steel_type=steel_type)
                dfs_by_steeltype[steel_type] = pd.concat([dfs_by_steeltype[steel_type], pd.DataFrame(augmented_data, columns=dfs_by_steeltype[steel_type].columns)], ignore_index=True)
            print("Augmented: ", steel_type)
        self.save_new_augmented_dataset(dfs_by_steeltype, data_path, data.columns)

    def group_by_steel_type(self, data):
        element_data_grouped_by_steeltype = data.groupby(data.iloc[:, -1])
        dfs_by_steeltype = {}

        for type, group_data in element_data_grouped_by_steeltype:
            dfs_by_steeltype[type] = group_data.iloc[:, :-1]
        return dfs_by_steeltype

    def calc_amounts(self, dfs_by_steeltype):
        amount_to_dict = {}
        for steel_type in dfs_by_steeltype.keys():
            if len(dfs_by_steeltype[steel_type]) < 300:
                amount_to_dict[steel_type] = 300 - len(dfs_by_steeltype[steel_type])
        return amount_to_dict

    def augment_for_steel_type(self, amount, data, steel_type):
        wgan_steel = WGAN_steel_type_data_augmentation()
        return wgan_steel.augment_data(amount, data, steel_type)

    def save_new_augmented_dataset(self, dfs_by_steeltype, data_path, column_labels):
        augmented_dataset = pd.DataFrame()
        for steel_type, data_frame in dfs_by_steeltype.items():
            steel_col = [steel_type for _ in range(len(data_frame))]
            data_frame[column_labels[-1]] = steel_col
            augmented_dataset = pd.concat([augmented_dataset, data_frame], ignore_index=True)
        augmented_dataset.columns = column_labels
        augmented_dataset.to_csv(data_path + "_with_augmented_data")

    def prepare_data(self, data):
        # make data numeric
        for column in data.columns:
            for index, value in data[column].items():
                if isinstance(value, str):
                    data.at[index, column] = value.replace(",", ".")
        return data