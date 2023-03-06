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
All of our .py files should be in one directory.  You may either call individual classifiers like so:

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

Or you can simply run compare.py.
