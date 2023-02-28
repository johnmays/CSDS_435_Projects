"""
Utility functions for CSDS 435 Project 1.
"""
import numpy as np
from typing import Tuple

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

def load_data(path: str) -> Tuple[np.ndarray, np.ndarray]: # (data, labels)
    """
    Loads classifier data from a CSV file.

    Args:
        path (str): the path to the CSV file.

    Returns:
        tuple: two values containing the data and the labels respectively.
    """
    return (None, None)


