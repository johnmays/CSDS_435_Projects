# How to run my code:
a brief instruction manual | John Mays | jkm100

## Environment:
You must have a python3 version with the following packages installed:
- numpy
- matplotlib
- re
- sklearn
- sklearn_extra
- scipy

I was running Python 3.9.13 with `venv`.  I will include my PIP list at the bottom in case there are package issues.

All of my .py files should be in one directory.
There should be:
- `util.py`, `generate_results.py`,`clustering_algo.py` (abstract class), `kmedoids.py`, `completelink.py`, `data/cnnhealth.txt`, `distance_matrics.npy` (pre-generated .npy file), `distance_matrix_1.txt`, and `distance_matrix_2.txt` (output files for assignment)
-------------
## Clustering and Getting my Results from Assignment
Call my `generate_results.py` file from the command line; you must specify the path to the data:

Here is an example call:
`python3 generate_results.py data/cnnhealth.txt`

This command alone will run both algos with both distance metrics and print out summarys of performance.(**PART OF THE RESULTS FOR PART D OF ASSIGNMENT**)

### You can add flags for additional functionality:

- `--regen` if you want to manually generate distance matrices instead of loading the ones I submitted
- `--tuning` will turn on `tuning` which will iterate through list of `k`'s and plot their performance. (**RESULTS FOR PART C OF ASSIGNMENT**)
- `--topics` when submitted, the program will print out the top tokens for the top clusters (**RESULTS FOR PART F OF ASSIGNMENT**)
- `--plots` will print the distributions of the cluster sizes (**PART OF THE RESULTS FOR PART D OF ASSIGNMENT**) as well as my distance matrix visualizations (**RESULTS FOR PART E OF ASSIGNMENT**)
- `--consistency` will run a method to use entropy and purity to compare two runs of different methods (**PART OF THE RESULTS FOR PART D OF ASSIGNMENT**)

One more example call:
`python3 generate_results.py data/cnnhealth.txt --topics`

-------------

## May be helpful to know:

### > If you want to load BOW data:
Call my `util.load_data()` like so:

`X,vocab = load_data('data/cnnhealth.txt')`

### > If you want to load (pre-generated) distance matrices:
Call my `util.load_distance_matrices()` like so:

`distance_matrix_1, distance_matrix_2 = load_distance_matrices()`

### > If you want to generate distance matrices yourself:
First, load BOW data(how-to mentioned above), then call `util.create_distance_matrices()` like so:

`distance_matrix_1, distance_matrix_2 = create_distance_matrices(X)`

#### Note: if you choose to do this, cnnhealth.txt should be in a directory named "data".  The "data" directory should be on the same level as all of the .py files.
-------------
PIP List:
appnope             0.1.3
asttokens           2.2.1
backcall            0.2.0
comm                0.1.3
contourpy           1.0.7
cycler              0.11.0
debugpy             1.6.6
decorator           5.1.1
executing           1.2.0
fonttools           4.39.3
importlib-metadata  6.1.0
importlib-resources 5.12.0
ipykernel           6.22.0
ipython             8.12.0
jedi                0.18.2
joblib              1.2.0
jupyter_client      8.1.0
jupyter_core        5.3.0
kiwisolver          1.4.4
matplotlib          3.7.1
matplotlib-inline   0.1.6
nest-asyncio        1.5.6
numpy               1.24.2
packaging           23.0
parso               0.8.3
pexpect             4.8.0
pickleshare         0.7.5
Pillow              9.5.0
pip                 23.0.1
platformdirs        3.2.0
prompt-toolkit      3.0.38
psutil              5.9.4
ptyprocess          0.7.0
pure-eval           0.2.2
Pygments            2.14.0
pyparsing           3.0.9
python-dateutil     2.8.2
pyzmq               25.0.2
regex               2023.3.23
scikit-learn        1.2.2
scikit-learn-extra  0.3.0
scipy               1.10.1
setuptools          67.4.0
six                 1.16.0
sklearn             0.0.post1
stack-data          0.6.2
threadpoolctl       3.1.0
tornado             6.2
traitlets           5.9.0
typing_extensions   4.5.0
wcwidth             0.2.6
wheel               0.38.4
zipp                3.15.0