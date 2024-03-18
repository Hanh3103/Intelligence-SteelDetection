# UMAP Algorithm 
Implemented by Marouane Soussi
## Overview

Uniform Manifold Approximation and Projection (UMAP) is a dimensionality reduction technique that aims to preserve both local and global structure in high-dimensional data. It is particularly effective in revealing non-linear relationships within the data, making it useful for tasks such as visualization and clustering.

## `apply_umap` Function

### Description

The `apply_umap` function takes a pandas DataFrame as input and applies UMAP to reduce the dimensionality of the numeric columns in the DataFrame. The resulting low-dimensional representation is returned.

### Parameters

- `data` (pandas DataFrame): Input data to be transformed.
- `n_neighbors` (int, default=15): The size of the local neighborhood used for manifold approximation.
- `min_dist` (float, default=0.1): The minimum distance between points in the low-dimensional representation.
- `n_components` (int, default=2): The number of dimensions in the embedded space.

### Returns

- `u` (numpy array): The low-dimensional representation of the input data.

### Usage

```python
import pandas as pd
from umap import UMAP

# Assuming 'your_data' is a pandas DataFrame
result = apply_umap(your_data, n_neighbors=15, min_dist=0.1, n_components=2)
```

### Note
- Input data must be a pandas DataFrame containing numeric columns.
- The UMAP algorithm is applied to the numeric columns of the DataFrame.
- The metric for UMAP can be specified (e.g., 'euclidean', 'cosine', etc.), with the default being 'euclidean'.

## UMAP Algorithm
UMAP works by constructing a high-dimensional topological representation of the data and then optimizing a low-dimensional representation to match it. The algorithm consists of the following steps:

Construct a High-Dimensional Representation:

1- Define a high-dimensional graph representing the local relationships between data points.
Use a specified metric (e.g., Euclidean distance) to measure distances between points.
Optimize Low-Dimensional Representation:

2- Create a low-dimensional representation of the data.
Optimize the representation to preserve both local and global structures.
Balancing Global and Local Structure:

3- UMAP aims to balance the preservation of local neighborhoods and global structures, providing a more flexible representation compared to other dimensionality reduction techniques.
Adjusting Parameters:

4- Parameters like n_neighbors, min_dist, and n_components allow users to control the trade-off between preserving local and global structures.

# Hierarchical Agglomerative Clustering Algorithm 
Implemented by Marouane Soussi
## Overview

Hierarchical Agglomerative Clustering is a clustering technique that builds a hierarchy of clusters. It starts with individual data points as separate clusters and successively merges them based on their similarity until the desired number of clusters is achieved.

## `hierarchical` Function

### Description

The `hierarchical` function applies Hierarchical Agglomerative Clustering to the input data and assigns each data point to a cluster. The resulting DataFrame includes an additional 'Cluster' column indicating the assigned cluster for each data point.

### Parameters

- `data` (pandas DataFrame): Input data for clustering.
- `n_clusters` (int): The number of clusters to form.

### Returns

- `data` (pandas DataFrame): The input data with an additional 'Cluster' column indicating the assigned cluster for each data point.

### Usage

```python
import pandas as pd
from sklearn.cluster import AgglomerativeClustering

# Assuming 'your_data' is a pandas DataFrame
result = hierarchical(your_data, n_clusters=3)
```

### Note
- Input data must be a pandas DataFrame containing numeric columns.
- The `n_clusters` parameter determines the number of clusters to form.
- The default linkage method is 'ward', and the default distance metric is 'euclidean', but these can be adjusted as needed.