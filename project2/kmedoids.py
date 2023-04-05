from clustering_algo import ClusteringAlgorithm

import numpy as np
from sklearn_extra.cluster import KMedoids

class kMedoidsAlgorithm(ClusteringAlgorithm):
    def __init__(self):
        self.clusterer = None

    @property
    def name(self):
        return "k-Medoids"
    
    def cluster(self, distance_matrix:np.ndarray, k:int) -> np.ndarray:
        self.clusterer = KMedoids(n_clusters=k, metric='precomputed')
        return self.clusterer.fit(distance_matrix).labels_
    
    def predict(self):
        pass

        