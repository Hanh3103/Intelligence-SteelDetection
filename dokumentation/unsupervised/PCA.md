
### PCA
Implemented by Joshua Kuhwald

Principal Component Analysis (PCA) is a method of unsupervised learning, in which the dimensionality of the data is reduced. The goal of PCA is to transform the data in such a way that the data in the new dimensions have as much variance as possible. PCA is applied in the context of the FoPra to visualize the data and to reduce the dimensionality of the data.

The method pca is implemented in the file MLModels/unsupervised/clustering_Algorithm.py.
The method pca takes a DataFrame and optionally a scaler as parameters.
The method returns an array that contains the explained variance ratio of all elements.

`def pca(dataframe, scaler=None):`

The method uses the PCA from sklearn.decomposition to transform the data.
Only the numerical columns of the DataFrame are used for the PCA.

Parameters:

DataFrame _dataframe_: The passed DataFrame must contain at least one numerical column.
This makes the method flexible, as it can be applied to different datasets with different numbers of elements.
Care should be taken to ensure that the passed DataFrame does not contain numerical columns that should not be included in the PCA.

Scaler _scaler_: The parameter _scaler_ is optional and can be used to scale the data column-wise.
Any scaler from sklearn.preprocessing can be used. A scaler can be used to normalize the data so that the data in the columns are comparable.
If no scaler is passed, the data will not be scaled.

Output: An array that contains the explained variance ratio of all elements.

Tests:
Unit tests for the method pca have been implemented, which can be found in the file tests/unsupervised/test_pca.py.
The unit tests check whether the method handles correct and incorrect inputs correctly and whether the method delivers the expected results.
Furthermore, the method pca was tested on two datasets, and the results were visualized (not included in the file, see Jupyter Notebook advanced_trials.ipynb).

Various other methods and implemented functions that were tried out in the context of the FoPra, but did not make it into the final version for various reasons, can be found in the Jupyter Notebook advanced_trials.ipynb.

