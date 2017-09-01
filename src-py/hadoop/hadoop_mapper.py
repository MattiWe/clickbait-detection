#!/usr/bin/python3
import sys
import json
import random

iterations_per_model = 500
sample_size = 1
for line in sys.stdin:
    # build subsets, pass the configuration to reducer
    __line = line.strip().split('\t')
    _line = int(__line[1])
    selected_indices_boolean_size = _line
    selected_indices_boolean_range = range(selected_indices_boolean_size)
    selected_indices_boolean = [1] * _line
    for i in range(iterations_per_model):
        index = random.sample(selected_indices_boolean_range, sample_size)
        for j in index:
            selected_indices_boolean[j] = 0
        keep_index = [x for x in selected_indices_boolean_range if selected_indices_boolean[x] == 1]
        print("{}\t{}\t{}".format(__line[0], json.dumps({"modelNumber": __line[0],
                                                         "runNumber": i,
                                                         "removedIndex": index,
                                                         "mse": 0}), json.dumps(keep_index)))
