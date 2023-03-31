"""
Utility functions for CSDS 435 Project 2.
"""
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt
import re

stopwords = ['', "wasn't", 'there', 'been', 'they', 'until', 'yourself', 'that', 'themselves', 'into', 'where', 'he', 'both', "it's", 'too', 'under', 'no', 'in', 'the', 'me', 'which', 'do', 'i', 'own', 'their', 'him', "that'll", 'up', '&amp;', 'same', 'at', 'a', 'rt', "aren't", 'who', 'on', 'we', 'being', 'can', 'against', 'your', 'she', 'again', 'whom', 'will', 'as', 'did', 'has', "isn't", 'few', 'through', 'down', "should've", "you'll", 'myself', 'himself', 'such', 'am', 'tip', 'most', 'but', 'while', 'to', 'if', "weren't", "won't", 'us', 'and', 'how', 'be', 'each', 'some', 'may', 'says', 'about', 'for', 'any', 'only', 'are', 'his', 'with', 'by', 'here', "doesn't", 'all', 'then', 'of', 'hers', 'them', 'or', 'theirs', 'because', 'having', 'itself', 'what', 'yourselves', "shouldn't", 'its', 'were', 'during', "didn't", 'further', 'nor', 'out', 'had', "you'd", 'not', 'you', 'could', 'our', 'my', 'other', 'herself', 'after', 'her', 'those', 'why', 'have', "hasn't", "you're", 'once', 'doing', 'below', 'yours', "she's", '--', 'is', 'off', 'an', 'via', "hadn't", 'from', 'ours', 'so', "wouldn't", "you've", 'this', 'over', 'it', 'between', 'when', "haven't", 'more', 'before', "couldn't", 'than', 'very', 'these', 'ourselves', "don't", 'above', 'does', 'was', 'should', 'rt', 'would', 'w/', 'just', '\r\n']

def load_data(path: str, stopwords = stopwords): 
    print('Importing data...')
    # format file into a list of tweets:
    lines = []
    with open(path, newline='\n') as file:
        lines = file.readlines()
    # take last column, take out punctuation, links, and hashtag characters, make lowercase, split into list of words
    for i, line in enumerate(lines):
        line = line.split('|')[2] 
        formatted_line = re.sub(r'!|:|,|\.|\||#', '', re.sub(r'http\S+', '', line))
        words_in_a_line = formatted_line.lower().split(' ')
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
    print("# Top 10 Legal Tokens: {}".format(vocab[np.flip(np.argsort(np.sum(X, axis=0)))[0:10]]))

def distance1(x1,x2):
    """
    unordered, normalized edit distance
    """
    return np.sum(np.abs(x1-x2))/(np.sum(x1)+np.sum(x2))

def distance2(x1,x2):
    """
    Euclidean distance
    """
    return np.sqrt(np.sum(np.power((x1-x2), 2)))

def create_distance_matrix(X:np.ndarray, distance_type=1) -> np.ndarray:
    assert distance_type in (1,2)
    if distance_type == 1:
        distance_measure = distance1
    else:
        distance_measure = distance2
    m,n = np.shape(X)
    distance_matrix = np.zeros(shape=(m,m))
    for i in range(m):
        print('progress: {}'.format(str(i/m)), end='\r')
        for j in range(i):
            distance_matrix[i,j] = distance_measure(X[i],X[j])
            distance_matrix[j,i] = distance_matrix[i,j]
    print('done with distance metric {}!'.format(str(distance_type)))
    return distance_matrix