from classifier import Classifier
from util import MNISTnormalize, get_matrix_dataset, encode_keras_predictions
# import tensorflow as tf
from keras import layers, models, losses
from tensorflow import data
import numpy as np

class ConvolutionalNN(Classifier):
    def __init__(self):
        self.net = None

    @property
    def name(self):
        return "Convolutional Neural Network"

    def train(self, training_data: np.ndarray, training_labels: np.ndarray) -> None:
        training_data = np.copy(training_data)
        # normalizing from [0-255] -> [0,1]:
        training_data = MNISTnormalize(training_data)
        # transforming data to a 3D tensor full of images:
        training_data = get_matrix_dataset(training_data)
        # putting it into a format acceptable by keras:
        # training_data = data.Dataset.from_tensor_slices((training_data, training_labels))
        training_data = training_data.reshape(np.shape(training_data)+(1,))
        self.net = models.Sequential()
        self.net.add(layers.Conv2D(32, (3,3), activation='relu', input_shape = (28,28,1)))
        self.net.add(layers.MaxPooling2D((2, 2)))
        self.net.add(layers.Conv2D(64, (3,3), activation='relu'))
        self.net.add(layers.MaxPooling2D((2, 2)))
        self.net.add(layers.Flatten())
        self.net.add(layers.Dense(16, activation='relu'))
        self.net.add(layers.Dense(10, activation='softmax'))
        print("Fitting Conv Net...")
        self.net.compile(optimizer='adam',
              loss=losses.SparseCategoricalCrossentropy(),
              metrics=[])
        # self.net.fit(training_data, epochs=1)
        self.net.fit(training_data,training_labels, epochs=10)
        print("Finished!")

    def infer(self, testing_data: np.ndarray) -> np.ndarray:
        # normalizing from [0-255] -> [0,1]:
        testing_data = MNISTnormalize(testing_data)
        # transforming data to a 3D tensor full of images:
        testing_data = get_matrix_dataset(testing_data)
        # putting it into a format acceptable by keras:
        testing_data = testing_data.reshape(np.shape(testing_data)+(1,))
        return encode_keras_predictions(self.net.predict(testing_data))