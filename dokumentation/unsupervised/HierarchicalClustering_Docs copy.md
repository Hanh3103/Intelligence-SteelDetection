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

## Hierarchical Agglomerative Clustering Algorithm
1. Initialize Clusters (Start with each data point as a separate cluster.)
2. Compute Pairwise Distances
3. Merge Closest Clusters (Merge the two closest clusters based on a specified linkage method (e.g., 'ward', 'complete', 'average').)
4. Repeat (Repeat steps 2 and 3 until the desired number of clusters is reached.)
5. Assign Cluster Labels (Assign each data point to a cluster based on the clustering results.)
6. Output (The output is a DataFrame with an additional 'Cluster' column indicating the assigned cluster for each data point.)


## About Notebook :
Implemented by Marouane Soussi

In the provided analysis notebook is an approach that merges the usage of both algorithms to be found. The approach is to cluster 
the data first and choose the best number of clusters using silhouette score (Grid Search approach). After that a pipeline is created to reduce and cluster to data and tune 
the reduction parameters of umap algorithm to find the model that fits the best silhouette score to the clustering in reduction mode. This approach can be used for
further implementations. A code snippet can be provided
```python
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from umap import UMAP

range_neighbors = [5,6,7,8,9,10,11,12,13,14,15,16,17,18,19,20]
range_mindist = [0.1, 0.2, 0.3, 0.4, 0.5,0.8,0.9]



def recommend_pars(data, range_neighbors, range_mindist):
    best_model = None
    best_score = -1
    for i in range_neighbors:
        for j in range_mindist:
            pipeline = Pipeline([
                ('umap', UMAP(n_neighbors=i,
                              min_dist=j,
                              n_components=2,
                              metric='euclidean')),
                ('clustering', AgglomerativeClustering(n_clusters=3))
            ])
            cluster_labels = pipeline.fit_predict(data)
            silhouette_score_pipeline = silhouette_score(data, cluster_labels)
            print(silhouette_score_pipeline)
            if silhouette_score_pipeline > best_score:
                best_score = silhouette_score_pipeline
                best_model = pipeline

    return best_model,best_score

best_model,best_score = recommend_pars(dataset, range_neighbors, range_mindist)

print("Best silhouette score:", best_score)
```