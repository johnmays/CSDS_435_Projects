# How to run my code:
Brief instruction manual

## Environment:
You must have a python3 version with the following packages installed:
- numpy
- matplotlib
- re


All of my .py files should be in one directory.

If you want to generate my distance matrices instead of just loading them, cnnhealth.txt should be in a directory named "data".  The "data" directory should be on the same level as all of the .py files.

## Basic Data Actions:

### > If you want to load BOW data:
Call my `util.load_data()` like so:

`X,vocab = load_data('data/cnnhealth.txt')`

### > If you want to load (pre-generated) distance matrices:
Call my `util.load_distance_matrices()` like so:

`distance_matrix_1, distance_matrix_2 = load_distance_matrices()`

### > If you want to generate distance matrices yourself:
First, load BOW data(how-to mentioned above), then call `util.create_distance_matrices()` like so:

`distance_matrix_1, distance_matrix_2 = create_distance_matrices(X)`

## Generating Report Elements:

### > If you want to generate summary of BOW data from my report:
First, load BOW data(how-to mentioned above), then call `util.generate_statistics()` like so:

`generate_statistics(X, vocab)`

### > If you want to generate the distance histograms from my report:
First, load the distance matrices(how-to mentioned above), then call `util.plot_distance_distributions()` like so:

`plot_distance_distributions(distance_matrix_1, distance_matrix_2)`

### > If you want to output the distance matrices to txt files OR a npy stash:
First, load the distance matrices(how-to mentioned above), then call `util.save_distance_matrices()` like so:

`save_distance_matrices(distance_matrix_1, distance_matrix_2, save_type='txt')` -> generates two text files

OR

`save_distance_matrices(distance_matrix_1, distance_matrix_2, save_type='npy')` -> generates one numpy stash file

