#!/usr/bin/python
import json
import sys
import random
import sklearn.metrics as skm
import scipy.sparse
import numpy as np
from ml import ClickbaitModel

# input comes from STDIN
for line in sys.stdin:
    # init variables once
    _line = json.loads(line)
    selected_indices_boolean = _line['selectedFeatures']
    n = 5
    cbm = ClickbaitModel()
    # load x, y from file
    x_train = scipy.sparse.load_npz('x_train.npz')
    x_test = scipy.sparse.load_npz('x_test.npz')
    y_train = np.load('y_train.npz')['data']
    y_test = np.load('y_test.npz')['data']

    # TODO repeat n times
    for i in range(n):
        index = random.choice(range(len(selected_indices_boolean)))
        while selected_indices_boolean[index] == 0:
            try:
                index += 1
            except IndexError:
                index = random.choice(range(len(selected_indices_boolean)))

        selected_indices_boolean[index] = 0
        # construct reduced matrices

        # remove columns
        x_train_current = x_train[:, [x for x in range(len(selected_indices_boolean)) if selected_indices_boolean[x] == 1]]
        x_test_current = x_test[:, [x for x in range(len(selected_indices_boolean)) if selected_indices_boolean[x] == 1]]
        # learn the model with reduced matrices

        cbm.regress(x_train_current, y_train, "RandomForestRegressor", evaluate=False)
        y_predicted = cbm.predict(x_test_current)

        current_mse = skm.mean_squared_error(y_predicted, y_test)

        # write to dict and print to stdout
        _line["runs"].append({"runNumber": i, "removedIndex": index, "mse": current_mse})

    _line['selectedFeatures'] = selected_indices_boolean
    print(json.dumps(_line))
