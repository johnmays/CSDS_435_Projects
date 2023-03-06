# How to run our code

## Environment:
You must have a python version with the following packages installed:
- Tensorflow
- Keras
- sklearn
- scipy
- numpy
- matplotlib
- time
- heapq
- tqdm

# Running Code:
All of our .py files should be in one directory.

training.csv should be in a directory named "data".  The "data" directory should be on the same level as all of the .py files.

You can simply run compare.py to output the three figures in our writeup and compare all the models.  By default, it will run 5-fold cross-validation split on all 6 models, but you can change that at the head of the file.  It takes about twenty minutes to run it all on my 2020 Macbook Pro.

You can also call individual classifiers like so, if you need to.

```
from naivebayes import NaiveBayes
import numpy as np

X, y = load_data('../data/training.csv')
data = cross_validation_split(X,y)
X_train, y_train, X_test, y_test = data[0]

nbayes = NaiveBayes()
nbayes.fit(X_train, y_train)

nbayes.infer(X_test)
```

predict.py is the file that outputs the CSV for the testing data.
