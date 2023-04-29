from predictor import Predictor

import numpy as np

class MFPredictor(Predictor):
    def __init__(self) -> None:
        pass

    @property
    def name(self):
        return "Matrix Factorization"
    
    def train(self, ratings:np.ndarray) -> bool:
        """
        train the recommender

        Args:
            ...
        """
        pass

    def predict(self, user, movie):
        """
        Get a rating for a new pair

        Args:
            ...
        """
        pass