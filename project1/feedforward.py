from classifier import Classifier
from sklearn.neural_network import MLPClassifier

import numpy as np

class FeedForwardNN(Classifier):
    def __init__(self):
        self.net = None

    @property
    def name(self):
        return "Feed-forward Neural Network"

    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        print("Fitting Feed Forward Net...")
        self.net = MLPClassifier(solver='lbfgs', alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1)
        self.net.fit(training_data, training_labels)
        print("Finished!")


    def infer(self, testing_data: np.ndarray) -> np.ndarray:
        pass