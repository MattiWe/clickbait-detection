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

mse_predict_mean = 0.062572763644
_mse = [mse_predict_mean - 0.037630503752052,
        mse_predict_mean - 0.05720698766876863,
        mse_predict_mean - 0.03690288911356149,
        mse_predict_mean - 0.04124547163960163,
        mse_predict_mean - 0.036660653468089506,
        mse_predict_mean - 0.03607672836723961,
        mse_predict_mean - 0.040758521386472416,
        mse_predict_mean - 0.040533835461634492,
        mse_predict_mean - 0.040313221699446081]
_lables = []

space = 0.1
lspace = 0.5
width = 0.5       # the width of the bars
ind_grey = [1]
ind_green = [1 + lspace + width, 1 + lspace + 2*width + space]
ind_purple = [1 + 2*lspace + 3*width + space,
              1 + 2*lspace + 4*width + 2*space,
              1 + 2*lspace + 5*width + 3*space]
ind_blue = [1 + 4*lspace + 5*width + 3*space,
            1 + 4*lspace + 6*width + 4*space,
            1 + 4*lspace + 7*width + 5  *space]

fig = plt.figure()
p2 = fig.add_subplot(111)
p2.bar(ind_grey, _mse[0], width, color='#303030')
p2.bar(ind_green, _mse[1:3], width, color='#83b458')
p2.bar(ind_purple, _mse[3:6], width, color='#984fbb')
p2.bar(ind_blue, _mse[6:], width, color='#44b4fe')
# blue '#44b4fe'
# green '#83b458'
# purple '#984fbb'
# rects2 = plt.bar(ind + width, women_means, width, color='y', yerr=women_std)
# p2.set_ylim(0.0)
p2.set_ylabel('MSE Difference to mean prediction')
p2.set_xticks(ind_grey + ind_green + ind_purple + ind_blue)
p2.set_xticklabels(('Ridge', "Lasso", 'RF', '', 'LSA', '', '', 'ANT', ''))
p2.set_title('Performance comparison between different techniques')
plt.show()

''' aco 5 pop 10 iter 5 hc w/ 10 replacements
(array([1, 0, 0, ..., 1, 1, 1]), 0.040758521386472416)
176165

aco 5 pop 30 iter 10 hc w/ 20 replacements
(array([0, 1, 0, ..., 1, 0, 1]), 0.040533835461634492)
175532

aco 5 pop 30 iter 10 hc w/ 100 replacements
(array([1, 0, 0, ..., 1, 0, 1]), 0.040313221699446081)
175512
'''

# add some text for labels, title and axes ticks

# plt.legend((rects1[0], rects2[0]), ('Men', 'Women'))
