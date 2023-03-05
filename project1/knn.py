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
        return "KNN"

    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        self.examples = training_data
        self.labels = training_labels
        print("Done preparing KNN!")

    def infer(self, testing_data: np.ndarray) -> np.ndarray:
        inferred_labels = []

        print("Predicting unseen samples with KNN...")
        for i in tqdm(range(len(testing_data))):
            instance = testing_data[i]
            knearest = []

            distances = np.linalg.norm(self.examples - instance, axis=1)
            kdistances = nsmallest(self.k, distances)
            for j in range(self.k):
                knearest.append(self.labels[np.where(distances == kdistances[j])[0][0]])
            
            inferred_labels.append(stats.mode(knearest)[0][0])

        return np.array(inferred_labels)
