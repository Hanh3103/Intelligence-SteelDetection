# Fuzzy C-Mean

implemented by Gabriel Marcelin

While this algorithm has been applied to the data, the output poses a certain difficulty. Therefore, you'll find an implementation of the algorithm, but it's not integrated into the final project.

The algorithm provides an alternative to k-means and has the advantage of implementing the principle of soft clustering. Each data point is assigned a probability for each cluster, unlike hard clustering, where there's a clear assignment.

For this algorithm, there are the `fuzzy_c_mean_fit(data, n_clusters)` and `fuzzy_c_mean_predict(data, cntr)` methods from the `skfuzzy` library.

## Preprocessing:

There is a Jupyter Notebook demonstrating possibilities for implementing the algorithm.
