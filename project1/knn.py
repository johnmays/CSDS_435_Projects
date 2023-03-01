from classifier import Classifier
from util import distance

from heapq import nsmallest
import numpy as np
from scipy import stats
from tqdm import tqdm

class KNearestNeighbor(Classifier):
    def __init__(self, k=5):
        self.k = k
        self.examples = None
        self.labels = None

    @property
    def name(self):
        return "K-Nearest Neighbors"

    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        self.examples = training_data
        self.labels = training_labels

    def infer(self, testing_data: np.ndarray) -> np.ndarray:
        inferred_labels = []

        for i in tqdm(range(len(testing_data))):
            instance = testing_data[i]
            knearest = []

            distances = np.zeros(len(self.examples))
            for j in range(len(self.examples)):
                distances[j] = distance(instance, self.examples[j])
            kdistances = nsmallest(self.k, distances)
            for j in range(len(self.examples)):
                if distances[j] in kdistances:
                    knearest.append(self.labels[j])
            
            inferred_labels.append(stats.mode(knearest)[0][0])

        return np.array(inferred_labels)
