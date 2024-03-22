### K-Means Clustering
Implemented by Joshua Kuhwald

K-means clustering is a method of unsupervised learning, in which the data is divided into k groups (clusters). The number of clusters k is predetermined. The goal of K-means clustering is to divide the data into the clusters in such a way that the data within a cluster is as similar as possible and the data between the clusters is as different as possible. The K-Means Clustering method was applied in the context of the FoPra to classify the steel samples into different classes.

The method kmeans is implemented in the file MLModels/unsupervised/cluster_Algorithm.py.
The method kmeans takes a DataFrame, the number of clusters _k_ and optionally a scaler as parameters.
The method extends the DataFrame by a column that contains the cluster membership of the data as int values _0 to k-1_.

`def kmeans(dataframe, k, scaler=None):`

The clustering uses the kmeans++ algorithm from sklearn.cluster.KMeans to divide the data into k clusters.
Only the numerical columns of the DataFrame are used for the clustering.

Parameters:

DataFrame _dataframe_: The passed DataFrame must contain at least one numerical column.

Integer _k_: The parameter _k_ can be freely chosen (must be int and _k>=2_) and specifies the number of clusters into which the data should be divided.

Scaler _scaler_: The parameter _scaler_ is optional and can be used to scale the data column-wise.
Any scaler from sklearn.preprocessing can be used. A scaler can be used to normalize the data so that the data in the columns are comparable.
If no scaler is passed, the data will not be scaled.

Output: the passed DataFrame with an additional column that contains the cluster membership of the data as int values _0 to k-1_.

Tests:
Unit tests for the method kmeans have been implemented, which can be found in the file tests/unsupervised/test_kmeans.py.
The unit tests check whether the method handles correct and incorrect inputs correctly and whether the method delivers the expected results.
Furthermore, the method kmeans was tested on two datasets, and the results were visualized (not included in the file, see Jupyter Notebook advanced_trials.ipynb).
One of the datasets is a dataset containing the actual steel types to validate the results of the K-Means Clustering.

Various other methods and implemented functions that were tried out in the context of the FoPra, but did not make it into the final version for various reasons, can be found in the Jupyter Notebook advanced_trials.ipynb.