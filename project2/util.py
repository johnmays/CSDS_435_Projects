"""
Utility functions for CSDS 435 Project 2.
"""
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import re

stopwords = ["a", "an", "the", "is", "to", "for", "in", "of", "and", "on", '']

def load_data(path: str, stopwords = ["a", "the", "is"]): 
    print('Importing data...')
    # format file into a list of tweets:
    lines = []
    with open(path, newline='\n') as file:
        lines = file.readlines()
    # take last column, take out punctuation, make lowercase, split into list of words
    for i, line in enumerate(lines):
        words_in_a_line = re.sub(r'!|:|,|\.|\|', '', line.split('|')[2]).lower().split(' ')
        lines[i] = remove_stopwords(words_in_a_line, stopwords)
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

def remove_stopwords(words, stopwords):
    return list(filter(lambda word: word not in stopwords, words))

def generate_statistics(X,vocab):
    print("Dataset Statistics:")
    m,n = np.shape(X)
    num_words = int(np.sum(X))
    print("# Number of Tweets: {}".format(m))
    print("# Number of Words (Total): {}".format(num_words))
    print("# Number of Tokens: {}".format(n))
    print("# Average Number of Words per Tweet: {}".format(round((num_words/m),3)))
    print("# Top 5 Legal Tokens: {}".format(vocab[np.flip(np.argsort(np.sum(X, axis=0)))[0:5]]))

def distance1(x1,x2):
    """
    nonlocal edit distance
    """
    return np.sum(np.abs(x1-x2))/np.sum(x1+x2)

def distance2(x1,x2):
    """
    Euclidean distance
    """
    return np.sqrt(np.sum(np.power((x1-x2), 2)))