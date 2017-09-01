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
from sklearn.linear_model import Ridge, Lasso
from sklearn.decomposition import SparsePCA, TruncatedSVD
import sklearn.metrics as skm
from copy import deepcopy
from collections import deque
import scipy.sparse
import json
import pickle
import os
import time
import matplotlib.pyplot as plt
import seaborn


def normalized_mean_squared_error(truth, predictions):
    norm = skm.mean_squared_error(truth, np.full(len(truth), np.mean(truth)))
    return skm.mean_squared_error(truth, predictions) / norm


# get the training data
cbd = ClickbaitDataset("../../clickbait17-validation-170630/instances.jsonl", "../../clickbait17-validation-170630/truth.jsonl")
cbm = ClickbaitModel()
ev_function = cbm.eval_regress
# f_builder = build_new_features(cbd)
f_builder = pickle.load(open("../feature_builder_w13_c00_2.pkl", "rb"))
x, x2, y, y2 = f_builder.build_features
print(x.shape)
print(x2.shape)
'''print("full set regression")
cbm.regress(x, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
# cbm.regress(x, y, Lasso(), evaluate=False)
y_predict = cbm.predict(x2)
ev_function(y2, y_predict)
print(skm.mean_squared_error(y2, np.full(len(y2), np.mean(y2))))
'''

'''
# A Decision Trees (ensemble techniques)
print("Random Forest")
cbm.regress(x, y, "RandomForestRegressor", evaluate=False)
y_predict = cbm.predict(x2)
ev_function(y2, y_predict)

# B PCA selection
print("tsvd regression")
'''
sv_list = [2, 5, 10, 15, 25, 50, 75, 100, 150, 200, 250, 300, 350, 400, 450, 500, 550, 600, 650, 700]
'''
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
        indices = run["removedIndex"]
        for i in indices:
            _index_to_mse_drop[i].append(run["mse"] - previous_mse)
        previous_mse = run["mse"]

_runnr_to_mse_drop_average = []
_index_to_mse_drop_average = deque()
_index_to_mse_drop_max = deque()
_index_to_mse_drop_min = deque()

print("compress matrices")
for r in _index_to_mse_drop:
    if r:
        r2 = np.asarray(r)
        _index_to_mse_drop_average.append(np.mean(r2))
        _index_to_mse_drop_max.append(np.amax(r2))
        _index_to_mse_drop_min.append(np.amin(r2))
    else:
        print("empty array at index {}".format(len(_index_to_mse_drop_average)-1))
        _index_to_mse_drop_average.append(0)
        _index_to_mse_drop_max.append(0)
        _index_to_mse_drop_min.append(0)

#_asdict = {x: (_index_to_mse_drop_min[x], _index_to_mse_drop_average[x], _index_to_mse_drop_max[x])
#           for x in range(len(_index_to_mse_drop_average))}
print("draw")
# _pos = np.linspace(0, (len(_index_to_mse_drop_average)+1)/1000, len(_index_to_mse_drop_average))
fig = plt.figure()
p2 = fig.add_subplot(111)
p2.plot(sorted(_index_to_mse_drop_max), label="maximum")
p2.plot(sorted(_index_to_mse_drop_min), label="minimum")
p2.plot(sorted(_index_to_mse_drop_average), label="average")
p2.set_ylabel('MSE Difference')
p2.set_xlabel('Feature Index')
p2.set_title('MSE Changes on Feature Removal')
p2.legend()
plt.show()

# TODO build reduced feature sets and train/predict
# 1. use highest mse
'''_mean_indices = sorted(range(len(_index_to_mse_drop_average)), key=lambda x: _index_to_mse_drop_average[x])
indices = _mean_indices[:100]
index_names = f_builder.feature_names
for i in indices:
    print(index_names[i])'''

'''i = 0.05
reduced_features = ([], [])
reduced_features_bad = []
while i > 0.001:
    feature_size = int(len(_index_to_mse_drop) * (i))
    # print("\n{} best features by mse drop mean".format(feature_size))
    indices = _mean_indices[-feature_size:]
    x_train_new = scipy.sparse.csc_matrix(x)[:, indices]
    x_test_new = scipy.sparse.csc_matrix(x2)[:, indices]
    # print("shapes: {}, {}".format(x_train_new.shape, x_test_new.shape))
    cbm.regress(x_train_new, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
    y_predict = cbm.predict(x_test_new)
    reduced_features[0].append(feature_size)
    reduced_features[1].append(skm.mean_squared_error(y2, y_predict))
    indices = _mean_indices[:feature_size]
    x_train_new = scipy.sparse.csc_matrix(x)[:, indices]
    x_test_new = scipy.sparse.csc_matrix(x2)[:, indices]
    # print("shapes: {}, {}".format(x_train_new.shape, x_test_new.shape))
    cbm.regress(x_train_new, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
    y_predict = cbm.predict(x_test_new)
    reduced_features_bad.append(skm.mean_squared_error(y2, y_predict))
    i = i - 0.002

fig = plt.figure()
p2 = fig.add_subplot(111)
p2.plot(reduced_features[0], reduced_features[1], label='best features')
# p2.plot(reduced_features[0], reduced_features_bad, label='worst features')
p2.set_ylabel('Mean Squared Error')
p2.set_xlabel('Nr. of features selected')
p2.set_title('MSE of Ridge on feature subsets selected by mean')
p2.legend()
plt.show()'''

# TODO find indices with high min/max spread -> candidates for interactions
