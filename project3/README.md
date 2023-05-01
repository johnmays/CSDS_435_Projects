# README
A brief instruction manual for CSDS 435 Project 3, Group 1.

John Mays | jkm100

Aaron Orenstein | aao62

## Included Files:
- `run.ipynb` a notebook where we run our code and present our results
- `util.py` a utilities file with helper methods and the methods for generating our results
- `neural_network.py`  a file that contains fitting & predicting methods for the neural network approach
- `matrix_factorization.py` a file that contains fitting & predicting methods for the matrix factorization` approach
- this README file
- `report.pdf` Our writeup.

## Environment:
You must have a python3 version with the following packages (and their dependencies) installed:
- re
- numpy
- matplotlib
- surprise
- keras
- ipykernel (for jupyter)

## Running Code:
The python files are meant to be where code is kept, and `run.ipynb` is where we have displayed its results *and* how to run it.  Some essential notes:
- data is imported from a remotely hosted repo, not a local one, and deterministically sorted to guarantee consistent cross-validation folds via a seeded algorithm: `util.fold()`
- Surprise's three builtin algorithms (Random, KNN, and SVD) are run via `util.run()`
- Due to the nature of data formatting, the matrix factorization and neural network algorithms are run with `util.run_mf()` and `util.run_nn()` respectively
- All three run methods return a `results` data structure that can be submitted to `util.display()` to be seen. In a fold-by-fold format.