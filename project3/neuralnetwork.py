from predictor import Predictor

import numpy as np
import keras

class NNPredictor(Predictor):
    def __init__(self) -> None:
        self.model = None

    @property
    def name(self):
        return "Neural Network"
    
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