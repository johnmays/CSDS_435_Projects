from classifier import Classifier

import numpy as np
from sklearn import svm

class SVM(Classifier):
    def __init__(self):
        self.SVM = None

    @property
    def name(self):
        return "Support Vector Machine"

    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        print("Fitting SVM...")
        self.SVM = svm.SVC(kernel="rbf")
        self.SVM.fit(training_data, training_labels)
        print("Finished!")

    def infer(self, testing_data: np.ndarray) -> np.ndarray:
        return self.SVM.predict(testing_data)