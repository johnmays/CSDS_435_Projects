"""
Utility functions for CSDS 435 Project 1.
"""
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import csv
from scipy import stats

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

def distance(x1: np.ndarray, x2: np.ndarray) -> float:
    """
    Get the Euclidian distance between two points.

    Args:
        x1 (np.ndarray): the first point
        x2 (np.ndarray): the second point

    Returns:
        float: the Euclidian distance between the points.
    """
    return np.linalg.norm(x1 - x2)

def entropy(labels: np.ndarray, partitions: np.ndarray) -> float:
    """
    Get the entropy of a node or set of child nodes.

    Args:
        labels (np.ndarray): the set of labels at this node/nodes.
        partitions (np.ndarray): if getting the weighted entropy of multiple
            child nodes, this is an array of the same length as 'labels' that
            represents the index of the partition for each label.
            Otherwise, this is None.
    
    Returns:
        float: the entropy of this node/nodes.
    """
    if partitions is None:
        counts = {}
        for label in labels:
            if label not in counts:
                counts[label] = 1
            else:
                counts[label] += 1
        
        h = 0
        for key in counts:
            p = counts[key] / len(labels)
            h += -p * np.log(p)
        
        return h
    else:
        h = 0

        num_indices = int(np.max(partitions) + 1)
        for i in range(num_indices):
            rows = np.where(partitions==i)
            partition = labels[rows]
            h += entropy(partition, None) * (len(partition) / len(labels))
        
        return h

def sample_data(X: np.ndarray, y: np.ndarray=None, p=0.10):
    '''
    Takes a random sample of the the data.
    '''
    if type(X) != np.ndarray or (type(y) != np.ndarray and y is not None):
        raise TypeError('X and y should be numpy arrays')
    if not((p <= 1.0) and (p >= 0.0)):
        raise ValueError('percentage should be in range [0,1]')
    num_examples = X.shape[0]
    num_samples = int(p*num_examples)
    rand_indices = np.random.choice(range(0,num_examples), size=num_samples, replace=False)
    if y is None:
        return X[rand_indices,:]
    else:
        return X[rand_indices,:], y[rand_indices]
    

def cross_validation_split(X, y, folds=5):
    if type(X) is not np.ndarray or type(y) is not np.ndarray:
        raise TypeError('X and y should be numpy arrays')
    num_examples = np.size(y)
    if type(folds) is not int or folds > num_examples or folds < 1:
        raise TypeError('folds should be a positive integer <= sample size')
    rand_indices = list(range(num_examples))
    np.random.shuffle(rand_indices)
    X_shuffled = X[rand_indices]
    y_shuffled = y[rand_indices]

    data = () # in form ((X_train,y_train,X_test,y_test),(X_...)...)
    for fold in (range(folds)):
        i = int((fold/folds)*num_examples)
        j = int(((fold+1)/folds)*num_examples)
        X_train = np.concatenate((X[0:i],X[j:]))
        y_train = np.concatenate((y[0:i],y[j:]))
        X_test = X[i:j]
        y_test = y[i:j]
        data += ((X_train, y_train, X_test, y_test),)
    return data