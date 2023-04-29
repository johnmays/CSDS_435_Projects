"""
Utility functions for CSDS 435 Project 3.
"""

from collections import namedtuple
import numpy as np
import matplotlib.pyplot as plt
from surprise import accuracy
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


Results = namedtuple("Results", "rmse mae")


def run(algo, data) -> Results:
    res = Results([], [])
    for trainset, testset in KFold(n_splits=5, random_state=666).split(data):
        # train and test algorithm.
        algo.fit(trainset)
        predictions = algo.test(testset)

        # Compute and print Root Mean Squared Error
        res.rmse.append(accuracy.rmse(predictions, verbose=False))
        res.mae.append(accuracy.mae(predictions, verbose=False))
    return res


def display(res: Results):
    rows = ["RMSE", "MAE"]
    cols = [""] + [f"Fold {i}" for i in range(len(res.rmse))]
    col_w = max([len(s) for s in rows + cols])
    cols = [c + " " * (col_w - len(c)) for c in cols]
    print(" |".join(cols))
    for r, d in zip(rows, [res.rmse, res.mae]):
        d = [r] + [str(int(d * 1e3) / 1e3) for d in d]
        d = [d + " " * (col_w - len(d)) for d in d]
        print(" |".join(d))
