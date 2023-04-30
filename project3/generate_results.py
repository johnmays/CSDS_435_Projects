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

from util import load_data, run, display
from surprise import Dataset
from surprise.model_selection import KFold

num_folds = 5

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
    parser.add_argument('--no-cv', dest='cv', action='store_true', help='Turns off five-fold cross-validation split')
    parser.set_defaults(tuning=False, cv=True)
    args = parser.parse_args()

    data_path = os.path.expanduser(args.path)
    tuning = args.tuning
    cv = args.cv

    # load data
    data = Dataset.load_builtin('ml-100k')

    # potentially split data
    folds = []
    kf = KFold(n_splits=num_folds)
    if cv:
        for trainset, testset in kf.split(data):
            folds.append((trainset, testset))
    else:
        kf = KFold(n_splits=num_folds)
        trainset, testset = next(kf.split(data)) # takes the first element from the generator
        folds.append((trainset, testset))
    
    # warnings.filterwarnings("ignore") # WILL SUPRESS WARNING FOR CLEARER OUTPUT
    for algorithm, name in zip(predictors, names):
        predictor = algorithm()
        print('================')
        print(f'{name}:')
        print('================')
        for i, fold in enumerate(folds):
            trainset, testset = fold
            # print statistics for this fold and this algo
            print(f'Fold {i+1}: ')
            # results = run(predictor, trainset)
            # display(results)

