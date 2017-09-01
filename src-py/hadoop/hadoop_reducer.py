#!/usr/bin/python3
import json
import sys
import random
import sklearn.metrics as skm
from sklearn.linear_model import Ridge
import scipy.sparse
import numpy as np
sys.path.append('.')

# load x, y from file
x_train_arrays = np.load("x_train.npz")
x_train = scipy.sparse.csc_matrix((x_train_arrays['data'], x_train_arrays['indices'], x_train_arrays['indptr']), shape=x_train_arrays['shape'])
x_test_arrays = np.load("x_test.npz")
x_test = scipy.sparse.csc_matrix((x_test_arrays['data'], x_test_arrays['indices'], x_test_arrays['indptr']), shape=x_test_arrays['shape'])
# x_train = scipy.sparse.load_npz('x_train.npz')
# x_test = scipy.sparse.load_npz('x_test.npz')
y_train = np.load('y_train.npz')['data']
y_test = np.load('y_test.npz')['data']
model = Ridge(alpha=3.5)

# input comes from STDIN
for line in sys.stdin:
    try:
        # init variables once
        __line = line.split('\t')
        _line = json.loads(__line[1])
        keep_index = json.loads(__line[2])
        # remove columns
        x_train_current = x_train[:, keep_index]
        x_test_current = x_test[:, keep_index]
        # learn the model with reduced matrices
        model.fit(x_train_current, y_train)
        y_predicted = model.predict(x_test_current)
        _line['mse'] = skm.mean_squared_error(y_predicted, y_test)

        sys.stderr.write("reporter:counter:iterations,complete,1\n")
        sys.stderr.flush()

        print(json.dumps(_line))
    except Exception as e:
        print(e)
