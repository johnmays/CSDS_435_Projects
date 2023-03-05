"""
Test and compare all six classifiers.
"""
import matplotlib.pyplot as plt
import time
from util import load_data, accuracy, cross_validation_split

from cnn import ConvolutionalNN
from decisiontree import DecisionTree
from feedforward import FeedForwardNN
from knn import KNearestNeighbor
from naivebayes import NaiveBayes
from svm import SVM


CLASSIFIERS = [ConvolutionalNN, DecisionTree, FeedForwardNN, \
               KNearestNeighbor, NaiveBayes, SVM]
NUM_FOLDS = 5

all_data, all_labels = load_data("data/training.csv")
all_folds = cross_validation_split(all_data, all_labels, folds=NUM_FOLDS)

names, accuracies, train_times, test_times = [], [], [], []
for c in CLASSIFIERS:
    avg_train_time, avg_test_time, avg_accuracy = 0, 0, 0

    for fold in all_folds:
        training_data, training_labels, testing_data, testing_labels = fold

        classifier = c()

        start_time = time.time()
        classifier.train(training_data, training_labels)
        train_time = time.time()
        y_predicted = classifier.infer(testing_data)
        test_time = time.time()

        avg_train_time += (train_time - start_time) / NUM_FOLDS
        avg_test_time += (test_time - train_time) / NUM_FOLDS
        avg_accuracy += accuracy(y_predicted, testing_labels) / NUM_FOLDS if y_predicted is not None else 0

    print(f"\n*** {classifier.name}, {NUM_FOLDS}-fold averages ***")
    print(f"Training duration: {round(avg_train_time, 3)}s")
    print(f"Testing duration: {round(avg_test_time, 3)}s")
    print(f"Accuracy: {avg_accuracy}")
    print("")

    # Record statistics to generate report figure
    names.append(classifier.name)
    accuracies.append(round(avg_accuracy, 3))
    train_times.append(round(avg_train_time, 3))
    test_times.append(round(avg_test_time, 3))

# Generate accuracy figure
plt.bar(names, accuracies)
for name, acc in zip(names, accuracies):
    plt.text(name, acc, acc, ha="center")
plt.ylabel("Testing Accuracy")
plt.title("Classifier Accuracies")
plt.show()

# Generate training time figure
plt.bar(names, train_times)
for name, tt in zip(names, train_times):
    plt.text(name, tt, tt, ha="center")
plt.ylabel("Time (s)")
plt.title("Classifier Training Time")
plt.show()

# Generate inference time figure
plt.bar(names, test_times)
for name, tt in zip(names, test_times):
    plt.text(name, tt, tt, ha="center")
plt.ylabel("Time (s)")
plt.title("Classifier Testing Time")
plt.show()