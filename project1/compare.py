"""
Test and compare all six classifiers.
"""
import matplotlib.pyplot as plt
import time
from util import load_data, accuracy

from cnn import ConvolutionalNN
from decisiontree import DecisionTree
from feedforward import FeedForwardNN
from knn import KNearestNeighbor
from naivebayes import NaiveBayes
from svm import SVM


CLASSIFIERS = [ConvolutionalNN, DecisionTree, FeedForwardNN, \
               KNearestNeighbor, NaiveBayes, SVM]

training_data, training_labels = load_data("data/training.csv")
testing_data, testing_labels = load_data("data/testing.csv")

#training_data, training_labels = training_data[100:], training_labels[100:]
#testing_data, testing_labels = training_data[:100], training_labels[:100]

names, accuracies = [], []
for c in CLASSIFIERS:
    classifier = c()

    start_time = time.time()
    classifier.train(training_data, training_labels)
    train_time = time.time()
    y_predicted = classifier.infer(testing_data)
    test_time = time.time()

    acc = accuracy(y_predicted, testing_labels) if y_predicted is not None else 0

    print(f"*** {classifier.name} ***")
    print(f"Training duration: {round(train_time - start_time, 3)}s")
    print(f"Testing duration: {round(test_time - train_time, 3)}s")
    print(f"Accuracy: {acc}")
    print("")

    # Record statistics to generate report figure
    names.append(classifier.name)
    accuracies.append(acc)

# Generate accuracy figure
plt.bar(names, accuracies)
for name, acc in zip(names, accuracies):
    plt.text(name, acc, acc, ha="center")
plt.ylabel("Testing Accuracy")
plt.title("Classifier Accuracies")
plt.show()