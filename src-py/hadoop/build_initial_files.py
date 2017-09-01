#!/usr/bin/python3
import scipy.sparse
import numpy as np
import sys
import json
sys.path.append('..')
from features import feature as ft
from features.feature_builder import FeatureBuilder
from features.ml import ClickbaitModel
from features.dataset import ClickbaitDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy
import pickle

# get list of scores and a list of the postTexts
cbd = ClickbaitDataset("../../clickbait17-validation-170630/instances.jsonl",
                       "../../clickbait17-validation-170630/truth.jsonl")

f_builder = pickle.load(open("../feature_builder_w_13_c13_reduced.pkl", "rb"))
x_train, x_test, y, y2 = f_builder.build_features_split

x_train = scipy.sparse.csc_matrix(x_train)
x_test = scipy.sparse.csc_matrix(x_test)
# save for hadoop task
# scipy.sparse.save_npz("x_train", x)
np.savez("x_train", data=x_train.data, indices=x_train.indices, indptr=x_train.indptr, shape=x_train.shape)
np.savez("y_train", data=y.data, shape=y.shape)
# scipy.sparse.save_npz("x_test", x2)
np.savez("x_test", data=x_test.data, indices=x_test.indices, indptr=x_test.indptr, shape=x_test.shape)
np.savez("y_test", data=y2.data, shape=y2.shape)


x_train_arrays = np.load("x_train.npz")
x_train = scipy.sparse.csc_matrix((x_train_arrays['data'], x_train_arrays['indices'], x_train_arrays['indptr']), shape=x_train_arrays['shape'])


with open("initial_feature_select.jsonl", 'w') as of:
    for i in range(500):
        of.write("{}\t{}".format(i, int(x_train.shape[1])))
        if i != 499:
            of.write("\n")
