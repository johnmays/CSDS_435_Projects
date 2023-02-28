"""
An abstract class representing a simple classifier.
"""
from abc import ABC, abstractmethod
import numpy as np

class Classifier(ABC):
    @property
    @abstractmethod
    def name(self):
        """
        Get the human-readable name for this class.
        """
        pass

    @abstractmethod
    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        """
        Train this classifier on a dataset.

        Args:
            training_data (np.ndarray): an array of training instances, one instance per row
            training_labels (np.ndarray): an array of training labels
        """
        pass

    @abstractmethod
    def infer(self, testing_data: np.ndarray) -> np.ndarray:
        """
        Infer the labels of instances using this classifier.

        Args:
            testing_data (np.ndarray): an array of testing instances, one instance per row

        Returns:
            np.ndarray: an array of predicted labels
        """
        pass