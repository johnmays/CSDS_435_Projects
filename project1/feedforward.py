from classifier import Classifier

import numpy as np

class FeedForwardNN(Classifier):
    def __init__(self):
        pass

    @property
    def name(self):
        return "Feed-forward Neural Network"

    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        pass

    def infer(self, testing_data: np.ndarray) -> np.ndarray:
        pass