#!/usr/bin/python3
import json
import sys
import random
import sklearn.metrics as skm
from sklearn.linear_model import Ridge
import scipy.sparse
import numpy as np
sys.path.append('.')

# input comes from STDIN
for line in sys.stdin:
    try:
        # init variables once
        __line = line.split('\t')
        _line = json.loads(__line[1])
        selected_indices_boolean_size = len(_line['selectedFeatures'])
        selected_indices_boolean_range = range(selected_indices_boolean_size)
        selected_indices_boolean = _line['selectedFeatures']
        n = 500
        # load x, y from file
        x_train_arrays = np.load("x_train.npz")
        x_train = scipy.sparse.csc_matrix((x_train_arrays['data'], x_train_arrays['indices'], x_train_arrays['indptr']), shape=x_train_arrays['shape'])
        x_test_arrays = np.load("x_test.npz")
        x_test = scipy.sparse.csc_matrix((x_test_arrays['data'], x_test_arrays['indices'], x_test_arrays['indptr']), shape=x_test_arrays['shape'])
        # x_train = scipy.sparse.load_npz('x_train.npz')
        # x_test = scipy.sparse.load_npz('x_test.npz')
        y_train = np.load('y_train.npz')['data']
        y_test = np.load('y_test.npz')['data']

        for i in range(n):
            index = random.sample(selected_indices_boolean_range, 10)

            for j in index:
                selected_indices_boolean[j] = 0
            # construct reduced matrices

            # remove columns
            x_train_current = x_train[:, [x for x in selected_indices_boolean_range if selected_indices_boolean[x] == 1]]
            x_test_current = x_test[:, [x for x in selected_indices_boolean_range if selected_indices_boolean[x] == 1]]
            # learn the model with reduced matrices
            model = Ridge(alpha=3.5, solver="sag")
            model.fit(x_train_current, y_train)
            y_predicted = model.predict(x_test_current)
            mse = skm.mean_squared_error(y_predicted, y_test)

            # write to dict and print to stdout
            _line["runs"].append({"runNumber": i, "removedIndex": index, "mse": mse})
            sys.stderr.write("reporter:counter:iterations,complete,1\n")
            sys.stderr.flush()

        _line['selectedFeatures'] = selected_indices_boolean
        print(json.dumps(_line))
    except Exception as e:
        print(e)
