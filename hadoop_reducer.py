#!/usr/bin/python

import json
import sys
import random

# input comes from STDIN
for line in sys.stdin:
    _line = json.loads(line)

    value = 0
    while not value:
        index = random.choice(range(len(_line['includedFeatures'])))
        value = _line['includedFeatures'][index]
    value = _line['includedFeatures'][index] = 0
    _line['mse'] = random.random()

    print(json.dumps(_line))
