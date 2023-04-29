"""
An abstract class for basic clustering algorithms.
"""
from abc import ABC, abstractmethod
import numpy as np

class Predictor(ABC):
    @property
    @abstractmethod
    def name(self):
        """
        Get the name for this class.
        """
        pass

    @abstractmethod
    def train(self, ratings:np.ndarray) -> bool:
        """
        train the recommender

        Args:
            ...
        """
        pass

    @abstractmethod
    def predict(self, user, movie):
        """
        Get a rating for a new pair

        Args:
            ...
        """
        pass