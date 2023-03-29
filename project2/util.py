"""
Utility functions for CSDS 435 Project 2.
"""
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt

def load_data(path: str): 
    print('Importing data...')
    # format file into a list of tweets:
    lines = []
    with open(path, newline='\n') as file:
        lines = file.readlines()
    # take the last column in a line (the tweet) and split it by spaces:
    for i, line in enumerate(lines):
        lines[i] = line.split('|')[2].split(' ')
    vocab = np.unique(np.concatenate(lines).flat)
    # creating data vector X:
    n = np.size(vocab)
    m = np.size(lines, axis=0)
    X = np.zeros((m,n))
    # populating X:
    for i, line in enumerate(lines):
        for word in line:
            j = np.where(vocab==word)[0]
            X[i,j] += 1

    print('X has {} examples and {} features in BOW format...'.format(m,n))
    return X, np.array(vocab)
