"""
The file for doing comparisons of algos.
"""
import argparse
import os.path
import warnings

import numpy as np
import matplotlib.pyplot as plt

purples = ["#0a0612", "#392249", "#482980", "#673ab7",
           "#7a52aa", "#9779bd", "#b59fd0", "#d3c5e3"]

from util import load_data

from surprise import NormalPredictor as Random
from surprise import SVD
from surprise import KNNBasic as KNN

from matfact import MFPredictor
from neuralnetwork import NNPredictor

predictors = [Random, SVD, KNN, MFPredictor, NNPredictor]
names = ['Random', 'SVD', 'KNN', 'Matrix Factorization', 'Neural Network']

if __name__ == '__main__':
    """
    Main method.  
    
    Parses args and generates results.
    """

    # Set up argparse arguments
    parser = argparse.ArgumentParser(description='Generate results for our clustering algorithms.')
    parser.add_argument('path', metavar='PATH', type=str, help='The path to the data(should be .txt).')
    parser.add_argument('--tuning', dest='tuning', action='store_true',
                        help='turns on tuning, which will loop through the algos many times to evaluate given hyperparameters.')
    parser.set_defaults(tuning=False)
    args = parser.parse_args()

    data_path = os.path.expanduser(args.path)
    tuning = args.tuning
    
    # load data

    # warnings.filterwarnings("ignore") # WILL SUPRESS WARNING FOR CLEARER OUTPUT
    for algorithm, name in zip(predictors, names):
        predictor = algorithm()
        print('================')
        print(f'{name}:')
        print('================')
        # print statistics for algo
