"""
Utility functions for CSDS 435 Project 3.
"""

from collections import namedtuple
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matrix_factorization import matrix_factorization
import time
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


Result = namedtuple("Result", "preds time")


# Run a surprise algorithm on the data
def run(algo, data) -> List[Result]:
    res = []
    for trainset, testset in fold(data):
        # train and test algorithm.
        t = time.time()
        algo.fit(trainset)
        res.append(Result(algo.test(testset), time.time() - t))
    return res


# Display the results of a surprise algorithm
def display(res: List[Result]):
    rows = ["RMSE", "MAE", "Time(s)"]
    data = [
        [foo(r.preds, verbose=False) for r in res]
        for foo in [accuracy.rmse, accuracy.mae]
    ] + [[r.time for r in res]]
    for d in data:
        d.append(sum(d) / len(d))
    cols = [""] + [f"Fold {i}" for i in range(len(res))] + ["Avg"]
    col_w = max([len(s) for s in rows + cols])
    cols = [c + " " * (col_w - len(c)) for c in cols]
    print(" |".join(cols))
    for r, d in zip(rows, data):
        d = [r] + [str(int(d * 1e3) / 1e3) for d in d]
        d = [d + " " * (col_w - len(d)) for d in d]
        print(" |".join(d))


# Convert data folds into lists of ratings
def get_xy(data, full_tr):
    res = []
    for trainset, testset in fold(data):
        ratings = trainset.build_testset()
        x_tr = [
            (full_tr.to_inner_uid(e[0]), full_tr.to_inner_iid(e[1])) for e in ratings
        ]
        y_tr = [e[2] for e in ratings]
        x_te = [
            (full_tr.to_inner_uid(e[0]), full_tr.to_inner_iid(e[1])) for e in testset
        ]
        y_te = [e[2] for e in testset]
        res.append((np.array(x_tr), np.array(y_tr), np.array(x_te), np.array(y_te)))
    return res


# Get predictions for test datasets on matrix factorizations
def run_mf(data, K, **kwargs) -> List[Result]:
    res = []
    for trainset, testset in fold(data):
        t = time.time()
        R = [[0] * trainset.n_items for _ in range(trainset.n_users)]
        for entry in trainset.all_ratings():
            R[entry[0]][entry[1]] = entry[2]

        P = np.random.rand(trainset.n_users, K)
        Q = np.random.rand(trainset.n_items, K)
        P, Q = matrix_factorization(R, P, Q, K, **kwargs)
        R = np.dot(P, Q.T)

        pred = []
        for uid, iid, r in testset:
            try:
                i_uid = trainset.to_inner_uid(uid)
                i_iid = trainset.to_inner_iid(iid)
                pred.append(Prediction(uid, iid, r, R[i_uid][i_iid], {}))
            except ValueError:
                pass
        res.append(Result(pred, time.time() - t))
    return res
