import pandas as pd
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import EditedNearestNeighbours
from sklearn.model_selection import train_test_split
from imblearn.combine import SMOTEENN


class Steel_Smote_Enn:
    """
    Class to perform SMOTE and Edited Nearest Neighbours (ENN) on steel type data for handling class imbalance.

    Attributes:
        data (DataFrame): The input data.
        steels (dict): Dictionary mapping steel types to unique numerical labels.
        X (DataFrame): Features extracted from the input data.
        y (Series): Target labels extracted from the input data.
    """

    def __init__(self, data):
        """
                Initializes the Steel_Smote_Enn class with the provided data.

                Args:
                    data (DataFrame): The input data containing features and target labels.
        """
        self.data = data
        self.steels = self.extract_steel_types()
        self.data.iloc[:, -1] = [self.steels[c] for c in self.data.iloc[:, -1]]
        self.data = self.data.map(pd.to_numeric)
        self.X = self.data.iloc[:, :-1]
        self.y = self.data.iloc[:, -1]

    def apply(self):
        """
               Applies SMOTE and ENN to the training data to handle class imbalance.

               Returns:
                   DataFrame: The resampled dataset with balanced classes.
        """
        X_train, X_test, y_train, y_test = train_test_split(self.X, self.y, test_size=0.2, random_state=42)

        smote = SMOTE(k_neighbors=2, random_state=42)
        enn = EditedNearestNeighbours(n_neighbors=5)

        smote_enn = SMOTEENN(smote=smote, enn=enn, random_state=42)

        X_resampled, y_resampled = smote_enn.fit_resample(X_train, y_train)
        reverse_decode_dict = {v: k for k, v in self.steels.items()}
        train_df  = pd.concat([X_resampled, y_resampled], axis=1)
        test_df  = pd.concat([X_test, y_test], axis=1)
        combined_df = pd.concat([train_df, test_df], axis=0).reset_index(drop=True)
        combined_df.iloc[:, -1] = [reverse_decode_dict[int(i)] for i in combined_df.iloc[:, -1]]
        combined_df.iloc[:, -1] = combined_df.iloc[:, -1].astype(str)
        return combined_df

    def extract_steel_types(self):
        """
        Extracts unique steel types from the last column of the DataFrame and assigns them unique numerical labels.

        Returns:
            dict: A dictionary containing unique steel types mapped to their numerical labels.
        """
        steels = {}
        counter = 0
        for i in range(len(self.data.iloc[:, -1])):
            if self.data.iloc[:, -1][i] not in steels:
                steels[self.data.iloc[:, -1][i]] = counter
                counter += 1
        return steels