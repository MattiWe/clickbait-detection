#!/usr/bin/python3
import sys
import numpy as np
sys.path.append('..')
import numpy as np
from features import feature as ft
from features.feature_builder import FeatureBuilder
from features.ml import ClickbaitModel
from features.dataset import ClickbaitDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.decomposition import SparsePCA, TruncatedSVD
from copy import deepcopy
from collections import deque
import scipy.sparse
import json
import pickle
import os
import time
import matplotlib.pyplot as plt

# get the training data
cbd = ClickbaitDataset("../../clickbait17-validation-170630/instances.jsonl", "../../clickbait17-validation-170630/truth.jsonl")
cbm = ClickbaitModel()
ev_function = cbm.eval_regress
# f_builder = build_new_features(cbd)
f_builder = pickle.load(open("../feature_builder.pkl", "rb"))
x, x2, y, y2 = f_builder.build_features

'''print("full set regression")
cbm.regress(x, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
y_predict = cbm.predict(x2)
ev_function(y2, y_predict)

# A Decision Trees (ensemble techniques)
print("Random Forest")
cbm.regress(x, y, "RandomForestRegressor", evaluate=False)
y_predict = cbm.predict(x2)
ev_function(y2, y_predict)'''

# B PCA selection
'''print("tsvd regression")
sv_list = [2, 5, 10, 15, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]

for v in sv_list:
    print(v)
    tsvd = TruncatedSVD(v)
    tsvd.fit(x)
    x_tsvd_dense = tsvd.transform(x)
    cbm.regress(x_tsvd_dense, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
    y_predict = cbm.predict(tsvd.transform(x2))
    ev_function(y2, y_predict)'''

# C own selection with cluster
# 1. concatenate all hadoop out files
print("read hadoop out files")
_result = []
out_files = os.listdir("../out")
for outfile in out_files:
    with open("../out/" + outfile, 'r') as of:
        _result += [json.loads(x) for x in of.readlines()]

# {"runs": [{"removedIndex", "runNumber", "mse"}], "selectedFeatures": [binary array]}
print("extract mse drop")
_index_to_mse_drop = [deque() for x in _result[0]["selectedFeatures"]]
for line in _result:
    previous_mse = line["runs"][0]["mse"]
    for run in line["runs"]:
        # higher > higher error
        _index_to_mse_drop[run["removedIndex"]].append(run["mse"] - previous_mse)
        previous_mse = run["mse"]

pickle.dump(_index_to_mse_drop, open("mse_drop.pkl", "wb"))
_runnr_to_mse_drop_average = []
_index_to_mse_drop_average = deque()
_index_to_mse_drop_max = deque()
_index_to_mse_drop_min = deque()

print("compress matrices")
for r in _index_to_mse_drop:
    r2 = np.asarray(r)
    _index_to_mse_drop_average.append(np.mean(r2))
    _index_to_mse_drop_max.append(np.amax(r2))
    _index_to_mse_drop_min.append(np.amin(r2))

fig = plt.figure()
p2 = fig.add_subplot(111)
p2.plot(_index_to_mse_drop_average)
p2.plot(_index_to_mse_drop_max)
p2.plot(_index_to_mse_drop_min)
plt.show()

# TODO find indices with high min/max spread -> candidates for interactions
