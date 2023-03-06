"""
Generate a CSV of predictions for testing data using our best classifier.
"""
from util import load_data

from cnn import ConvolutionalNN

best_classifier = ConvolutionalNN
training_data, training_labels = load_data("data/training.csv")
testing_data = load_data("data/testing.csv")

classifier = best_classifier()
classifier.train(training_data, training_labels)
predicted_labels = classifier.infer(testing_data)

with open("predictions.csv", "w") as file:
    file.write("labels\n")
    for label in predicted_labels:
        file.write(str(label) + "\n")
