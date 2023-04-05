"""
An abstract class for basic clustering algorithms.
"""
from abc import ABC, abstractmethod
import numpy as np

class ClusteringAlgorithm(ABC):
    @property
    @abstractmethod
    def name(self):
        """
        Get the name for this class.
        """
        pass

    @abstractmethod
    def cluster(self, distance_matrix:np.ndarray) -> np.ndarray:
        """
        Complete clustering process using provided distance matrix.

        Args:
            distance_matrix (np.ndarray): an nxn distance matrix for the dataset X.
        """
        pass