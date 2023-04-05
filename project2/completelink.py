from clustering_algo import ClusteringAlgorithm

import numpy as np
from sklearn.cluster import AgglomerativeClustering

class CompleteLinkAlgorithm(ClusteringAlgorithm):
    def __init__(self):
        self.clusterer = None

    @property
    def name(self):
        return "Complete Link Hierarchical Clustering"
    
    def cluster(self, distance_matrix:np.ndarray, k:int) -> np.ndarray:
        self.clusterer = AgglomerativeClustering(n_clusters=k, metric='precomputed', linkage='complete')
        return self.clusterer.fit(distance_matrix).labels_
    
    def predict(self):
        pass

        