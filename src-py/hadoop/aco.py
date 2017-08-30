#!/usr/bin/python3
import sys
import numpy as np
sys.path.append('..')
import numpy as np
import sklearn.metrics as skm
from features.feature_builder import FeatureBuilder
from features.ml import ClickbaitModel
from sklearn.linear_model import Ridge
from copy import deepcopy
from collections import deque
import scipy.sparse
import json
import pickle
import os
import time
import matplotlib.pyplot as plt
import random

cbm = ClickbaitModel()
ev_function = cbm.eval_regress
f_builder = pickle.load(open("../feature_builder_w13_c00.pkl", "rb"))
x, x2, y, y2 = f_builder.build_features

f_range = x.shape[1]
ph = [1] * f_range
best_solution = (None, 0)
max_ph = 0


# pass binary array with 1 == feature selected
def fitness(_x):
    x_train_current = x[:, [x for x in range(f_range) if _x[x] == 1]]
    x_test_current = x2[:, [x for x in range(f_range) if _x[x] == 1]]
    cbm.regress(x_train_current, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
    return skm.mean_squared_error(cbm.predict(x_test_current), y2)


def update_pheromones():
    global best_solution
    l_rate = 15
    f_rate = 0.1
    for index in range(f_range):
        # forget
        ph[index] = (1 - f_rate) * ph[index] + f_rate
        if best_solution[0][index] == 1:
            ph[index] = (1 - l_rate) * ph[index] + l_rate * best_solution[1]
    max_ph = max(ph)


def hillclimb(__x):
    global best_solution
    fitn = fitness(__x)
    for j in range(10):
        _x = deepcopy(__x)
        indices = random.sample(range(f_range), k=100)
        for i in indices:
            if _x[i] == 1:
                _x[i] = 0
            else:
                _x[i] = 1
        fitnew = fitness(_x)
        if fitnew < fitn:
            __x = _x
            fitn = fitnew
            if fitn < best_solution[1] or best_solution[1] == 0:
                best_solution = (_x, fitn)
    return __x


def build_pop():
    new_pop = [0] * f_range
    for p in range(f_range):
        if ph[p] > 0.3 * max_ph or random.random() < 0.2:
            new_pop[p] = 1
    return new_pop


def run():
    popsize = 5
    iterations = 30
    population = [np.random.randint(2, size=f_range) for i in range(popsize)]

    for i in range(iterations):
        print(i)
        for pop in population:
            pop = hillclimb(pop)
        update_pheromones()
    population = [build_pop() for x in range(popsize)]


run()
print(best_solution)
print(np.sum(best_solution[0]))
