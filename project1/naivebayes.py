from classifier import Classifier
from util import MNISTnormalize
from sklearn.naive_bayes import GaussianNB

import numpy as np

class NaiveBayes(Classifier):
    def __init__(self):
        self.model = None

    @property
    def name(self):
        return "Naive Bayes"

    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        training_data = np.copy(training_data)
        training_data = MNISTnormalize(training_data)

        print("Fitting NBayes...")
        self.model = GaussianNB()
        self.model.fit(training_data, training_labels)
        print("Finished!")

    def infer(self, testing_data: np.ndarray) -> np.ndarray:
        return self.model.predict(testing_data)