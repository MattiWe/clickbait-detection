#!/usr/bin/python

import json
import sys
import random

# input comes from STDIN
for line in sys.stdin:
    # loop over this a certain number of times and print changes over nr. of steps
    _line = json.loads(line)
    selected_indices_boolean = _line['includedFeatures']

    # load x, y from file
    x_train = scipy.sparse.load_npz('x_train.npz')
    x_test = scipy.sparse.load_npz('x_test.npz')
    y_train = np.load('y_train.npz')['data']
    y_test = np.load('y_test.npz')['data']

    value = 0
    while not value:
        index = random.choice(range(len(selected_indices_boolean)))
        value = selected_indices_boolean[index]
    selected_indices_boolean[index] = 0


    # construct reduced matrices


    # learn the model with reduced matrices


    # write to dict and print to stdout
    _line['mse'] = random.random()
    print(json.dumps(_line))
