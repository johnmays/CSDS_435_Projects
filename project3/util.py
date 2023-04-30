"""
Utility functions for CSDS 435 Project 3.
"""

from collections import namedtuple
from typing import List
import numpy as np
import matplotlib.pyplot as plt
from matrix_factorization import matrix_factorization
import time, math
from surprise import accuracy
from surprise.prediction_algorithms.predictions import Prediction
from surprise.model_selection import KFold

from neural_network import create_model


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


def get_data(res: List[Result]):
    data = [
        [foo(r.preds, verbose=False) for r in res]
        for foo in [accuracy.rmse, accuracy.mae]
    ] + [[r.time for r in res]]
    for d in data:
        d.append(sum(d) / len(d))
    return data


def get_avg(res: List[Result]):
    return [d[-1] for d in get_data(res)]


# Display the results of a surprise algorithm
def display(data, cols=None):
    rows = ["RMSE", "MAE", "Time(s)"]
    if cols is None:
        cols = [f"Fold {i}" for i in range(len(data))] + ["Avg"]
        data = get_data(data)
    cols = [""] + cols
    col_w = max([len(s) for s in rows + cols])
    cols = [c + " " * (col_w - len(c)) for c in cols]
    print(" |".join(cols))
    for r, d in zip(rows, data):
        d = [r] + [str(int(d * 1e3) / 1e3) for d in d]
        d = [d + " " * (col_w - len(d)) for d in d]
        print(" |".join(d))


def display_all(res, cols):
    data = [get_avg(r) for r in res]
    display([[d[i] for d in data] for i in range(3)], cols=cols)


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


# Train & get predictions fo
def run_nn(data, **kwargs) -> List[Result]:
    res = []
    full_tr = data.build_full_trainset()
    folds = get_xy(data, full_tr) # importing under special format for NN
    
    for fold in folds:
        X_train,y_train,X_test,y_test = fold
        # reshape:
        X_train = [X_train[:, 0], X_train[:, 1]]
        X_test = [X_test[:, 0], X_test[:, 1]]
        max_users = np.max([np.max(X_train[0]), np.max(X_test[0])])+1
        max_movies = np.max([np.max(X_train[1]), np.max(X_test[1])])+1
        # fitting:
        num_factors =50 # (hyperparameter)
        model = create_model(max_users, max_movies, num_factors)
        model.fit(x=X_train, y=y_train, batch_size=64, epochs=5, verbose=1) # validation_data=(X_test, y_test)
        # predicting:
        t = time.time()
        y_pred = model.predict(x=X_test)

        pred = []
        for i in range(y_test.size):
            uid = X_test[0][i]
            iid = X_test[1][i]
            r_uid = full_tr.to_raw_uid(uid)
            r_iid = full_tr.to_raw_iid(iid)
            pred.append(Prediction(r_uid, r_iid, y_test[i], y_pred[i], {}))
        res.append(Result(pred, time.time()-t))
    return res


# Compare
def compare(p_ratings: List[Prediction], p_truth: List[Prediction]):
    p_dict = {}
    for p in p_truth:
        arr = p_dict.get(p.uid)
        if arr is None:
            p_dict[p.uid] = {p.iid: p.est}
        else:
            arr[p.iid] = p.est

    rmse = mae = n = 0
    for p in p_ratings:
        r2 = p.est
        r1 = p_dict.get(p.uid, {}).get(p.iid)
        if not r1 is None:
            rmse += pow(r2 - r1, 2)
            mae += abs(r2 - r1)
            n += 1
    rmse = math.sqrt(rmse / n)
    mae /= n
    return rmse, mae


def compare_all(names, predictions):
    col_w = max(5, max([len(n) for n in names]))
    names = [n + " " * (col_w - len(n)) for n in names]
    prec = 1e3
    rmse = [[" " * col_w] * len(names) for _ in names]
    mae = [[" " * col_w] * len(names) for _ in names]
    for i, ps1 in enumerate(predictions):
        for j, ps2 in enumerate(predictions[i + 1 :]):
            j += i + 1
            rmse_ij, mae_ij = compare(ps1, ps2)
            s = str(int(rmse_ij * prec) / prec)
            rmse[i][j] = rmse[j][i] = s + " " * (col_w - len(s))
            s = str(int(mae_ij * prec) / prec)
            mae[i][j] = mae[j][i] = s + " " * (col_w - len(s))
    print(" |".join(["RMSE" + " " * (col_w - 4)] + names))
    for n, row in zip(names, rmse):
        print(" |".join([n] + row))
    print()
    print(" |".join(["MAE" + " " * (col_w - 3)] + names))
    for n, row in zip(names, mae):
        print(" |".join([n] + row))
