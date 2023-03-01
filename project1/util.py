"""
Utility functions for CSDS 435 Project 1.
"""
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import csv

def accuracy(y_predicted: np.ndarray, y: np.ndarray) -> float:
    """
    Gets the accuracy of an array of labels.

    Args:
        y_predicted (np.ndarray): an array of predicted labels.
        y (np.ndarray): the true labels to compare against.

    Returns:
        float: the accuracy of the predicted labels, in the range [0, 1].
    """
    if len(y_predicted) != len(y):
        raise ValueError(f"Array to compute accuracy of has a mismatched number of \
                         predicted labels (expected {len(y)} but got {len(y_predicted)})")
    return np.sum(y_predicted == y) / len(y)

def load_data(path: str):
    """
    Uses CSV header to determine if CSV contains MNIST train or test set.  Loads data in numpy data structures.

    Args:
        path (str): the path to the CSV file.

    Returns:
        X: numpy matrix of size (examples, attributes)
        y (OPTIONAL): numpy vector of size (examples,)
    """
    with open(path, newline='\n') as file:
        MNISTreader = csv.reader(file, delimiter=',')
        num_rows = 0
        num_cols = 0
        training = True
        for i, row in enumerate(MNISTreader):
            if i == 0:
                if row[0] != 'label':
                    training = False
                    num_cols = len(row)
                else:
                    num_cols = len(row) -1
            else:
                num_rows = i
        print('Importing data...')
        print('Size of dataset: {} examples with {} attributes.\nTraining set: {}'.format(num_rows, num_cols, training))
        X = np.zeros((num_rows, num_cols))
        y = np.zeros((num_rows,))
    # you have to read again:
    with open(path, newline='\n') as file:
        MNISTreader = csv.reader(file, delimiter=',')
        for i, row in enumerate(MNISTreader):
            if i != 0:
                if training:
                    X[i-1] = np.array(row[1:]).astype(int)
                    y[i-1] = np.array(row[0]).astype(int)
                else:
                    X[i-1] = np.array(row).astype(int)
        if training:
            return X, y
        else:
            return X
        
def MNISTnormalize(X: np.ndarray) -> np.ndarray:
    '''
    Normalizes MNIST image data to range [0.0,1.0]
    '''
    return X/255

def vector2matrix(x: np.ndarray) -> np.ndarray:
    '''
    Changes a 1d example vector into a 2d matrix (going from (784,)->(28,28))

    Arguments:
        x: 1d (shape = (784,)) numpy array
    
    Returns:
        x_img: 2d(shape = (28,28)) numpy array
    '''
    x_img = np.zeros((28,28))
    for i in range(28):
        x_img[i] = x[(28*i):((28*(i+1)))]
    # for i in range(len(x)):
    #     x_img[int(i/28),(i%28)] = x[i]
    return x_img

def print_digit(x: np.ndarray):
    '''
    Prints MNIST image data so you can see it.

    Arguments:
        x: 1d (shape = (784,)) numpy array
    '''
    x_img = vector2matrix(x)
    plt.imshow(x_img, cmap='gray_r')