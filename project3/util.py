"""
Utility functions for CSDS 435 Project 3.
"""

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from matrix_factorization import matrix_factorization
from surprise import accuracy
from surprise.prediction_algorithms.predictions import Prediction
from surprise.model_selection import KFold

purples = [
    "#0a0612",
    "#392249",
    "#482980",
    "#673ab7",
    "#7a52aa",
    "#9779bd",
    "#b59fd0",
    "#d3c5e3",
]


def load_data(path: str):
    pass


def fold(data):
    return KFold(n_splits=5, random_state=666).split(data)


Index = namedtuple("Index", "uid iid")


# Run a surprise algorithm on the data
def run(algo, data):
    res = []
    for trainset, testset in fold(data):
        # train and test algorithm.
        algo.fit(trainset)
        res.append(algo.test(testset))
    return res


# Display the results of a surprise algorithm
def display(res):
    rows = ["RMSE", "MAE"]
    data = [
        [foo(r, verbose=False) for r in res] for foo in [accuracy.rmse, accuracy.mae]
    ]
    cols = [""] + [f"Fold {i}" for i in range(len(res))]
    col_w = max([len(s) for s in rows + cols])
    cols = [c + " " * (col_w - len(c)) for c in cols]
    print(" |".join(cols))
    for r, d in zip(rows, data):
        d = [r] + [str(int(d * 1e3) / 1e3) for d in d]
        d = [d + " " * (col_w - len(d)) for d in d]
        print(" |".join(d))


# Convert data folds into rating matrices
def get_fold_matrices(data):
    Rs = []
    for trainset, testset in fold(data):
        R = [[0] * trainset.n_items for _ in range(trainset.n_users)]
        for entry in trainset.all_ratings():
            R[entry[0]][entry[1]] = entry[2]
        Rs.append(R)
    return Rs


# Get predictions for test datasets on matrix factorizations
def run_mf(data, K, **kwargs):
    Rs = get_fold_matrices(data)
    Ps = [np.random.rand(len(R), K) for R in Rs]
    Qs = [np.random.rand(len(R[0]), K) for R in Rs]

    for i in range(len(Rs)):
        Ps[i], Qs[i] = matrix_factorization(Rs[i], Ps[i], Qs[i], K, **kwargs)
    Rs = [np.dot(P, Q.T) for P, Q in zip(Ps, Qs)]

    res = []
    for R, (trainset, testset) in zip(Rs, fold(data)):
        pred = []
        for uid, iid, r in testset:
            try:
                i_uid = trainset.to_inner_uid(uid)
                i_iid = trainset.to_inner_iid(iid)
                pred.append(Prediction(uid, iid, r, R[i_uid][i_iid], {}))
            except ValueError:
                pass
        res.append(pred)
    return res
