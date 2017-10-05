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
from sklearn.model_selection import train_test_split
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


def plot_compare_techniques(x, x2, y, y2, subset_indices=None):
    if subset_indices is not None:
        x = x[:, subset_indices]
        x2 = x2[:, subset_indices]
    print("Ridge")
    cbm.regress(x, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
    # cbm.regress(x, y, Lasso(), evaluate=False)
    y_predict = cbm.predict(x2)
    ridge_score = skm.mean_squared_error(y2, y_predict)

    print("Lasso")
    cbm.regress(x, y, "Lasso", evaluate=False)
    # cbm.regress(x, y, Lasso(), evaluate=False)
    y_predict = cbm.predict(x2)
    lasso_score = skm.mean_squared_error(y2, y_predict)

    # A Decision Trees (ensemble techniques)
    print("Random Forest")
    cbm.regress(x, y, "RandomForestRegressor", evaluate=False)
    y_predict = cbm.predict(x2)
    rf_score = skm.mean_squared_error(y2, y_predict)

    # B PCA selection
    print("lsa")
    tsvd = TruncatedSVD(150)
    tsvd.fit(x)
    x_tsvd_dense = tsvd.transform(x)
    cbm.regress(x_tsvd_dense, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
    y_predict = cbm.predict(tsvd.transform(x2))
    lsa_score_150 = skm.mean_squared_error(y2, y_predict)

    tsvd = TruncatedSVD(300)
    tsvd.fit(x)
    x_tsvd_dense = tsvd.transform(x)
    cbm.regress(x_tsvd_dense, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
    y_predict = cbm.predict(tsvd.transform(x2))
    lsa_score_300 = skm.mean_squared_error(y2, y_predict)

    tsvd = TruncatedSVD(600)
    tsvd.fit(x)
    x_tsvd_dense = tsvd.transform(x)
    cbm.regress(x_tsvd_dense, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
    y_predict = cbm.predict(tsvd.transform(x2))
    lsa_score_600 = skm.mean_squared_error(y2, y_predict)

    '''
    Ridge
    0.03285209086622662
    Lasso
    0.0572069876688
    Random Forest
    0.0374444797409
    lsa
    0.0410244287633
    0.034023307002
    0.0324427885481'''

    mse_predict_mean = skm.mean_squared_error(y2, np.full(len(y2), np.mean(y2)))
    _mse = [ridge_score,
            lasso_score,
            rf_score,
            lsa_score_150,
            lsa_score_300,
            lsa_score_600]
    _lables = []

    space = 0.1
    lspace = 0.5
    width = 0.5
    ind_grey = [1]
    ind_green = [1 + lspace + width, 1 + lspace + 2*width + space]
    ind_purple = [1 + 2*lspace + 3*width + space,
                  1 + 2*lspace + 4*width + 2*space,
                  1 + 2*lspace + 5*width + 3*space]
    fig = plt.figure()
    p2 = fig.add_subplot(111)
    p2.bar(ind_grey, _mse[0], width, color='#303030')
    p2.bar(ind_green, _mse[1:3], width, color='#83b458')
    p2.bar(ind_purple, _mse[3:], width, color='#984fbb')
    p2.set_ylabel('Prediction Error (MSE)')
    p2.set_xticks(ind_grey + ind_green + ind_purple)
    p2.set_xticklabels(('Ridge', "Lasso", 'RF', '', 'LSA', ''))
    # p2.set_title('Performance comparison between different techniques')
    plt.show()


def reconstruct_results():
    print("read hadoop out files")
    _result = []
    out_files = os.listdir("../out_different_splits")
    for outfile in out_files:
        with open("../out_different_splits/" + outfile, 'r') as of:
            _result += [json.loads(x) for x in of.readlines()]

    print("extract mse drop")
    _index_to_mse_drop = [deque() for i in range(x.shape[1])]
    # {"removedIndex": [8207], "runNumber": 9, "mse": 0.03374654598867049, "modelNumber": "99"}
    _models = [[[0, 0] for i in range(1001)] for j in range(1001)]
    for line in _result:
        if line['runNumber'] != 0:
            _models[int(line['modelNumber'])][line['runNumber']][0] = line['removedIndex'][0]
            _models[int(line['modelNumber'])][line['runNumber']][1] = line['mse']
        else:
            _models[int(line['modelNumber'])][line['runNumber']][0] = None
            _models[int(line['modelNumber'])][line['runNumber']][1] = line['mse']

    for model in _models:
        for run in range(len(model)):
            if run == 0:
                previous_mse = model[run][1]
            else:
                _index_to_mse_drop[model[run][0]].append(model[run][1] - previous_mse)
                previous_mse = model[run][1]

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
    return np.asarray(_index_to_mse_drop), list(_index_to_mse_drop_average), np.asarray(_index_to_mse_drop_min), np.asarray(_index_to_mse_drop_max)


def plot_mse_reduced_sets(_index_to_mse_drop, _index_to_mse_drop_average, x, x2, y, y2, add_bad_feats=False):
    # percent of best cutoff
    _mean_indices = sorted(range(len(_index_to_mse_drop_average)), key=lambda x: _index_to_mse_drop_average[x])
    # i = 0.05
    i = 1.0
    reduced_features = ([], [])
    reduced_features_bad = []
    min_values = [None, None]
    while i >= 0.001:
        feature_size = int(len(_index_to_mse_drop) * (i))
        # print("\n{} best features by mse drop mean".format(feature_size))
        indices = _mean_indices[-feature_size:]
        x_train_new = scipy.sparse.csc_matrix(x)[:, indices]
        x_test_new = scipy.sparse.csc_matrix(x2)[:, indices]
        # print("shapes: {}, {}".format(x_train_new.shape, x_test_new.shape))
        cbm.regress(x_train_new, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
        y_predict = cbm.predict(x_test_new)
        current_mse = skm.mean_squared_error(y2, y_predict)
        reduced_features[0].append(feature_size)
        reduced_features[1].append(current_mse)

        if min_values[0] is None or min_values[0] > current_mse:
            min_values[0] = current_mse
            min_values[1] = indices

        if add_bad_feats:
            indices = _mean_indices[:feature_size]
            x_train_new = scipy.sparse.csc_matrix(x)[:, indices]
            x_test_new = scipy.sparse.csc_matrix(x2)[:, indices]
            cbm.regress(x_train_new, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
            y_predict = cbm.predict(x_test_new)
            reduced_features_bad.append(skm.mean_squared_error(y2, y_predict))
        if i <= 0.02:
            i = i - 0.005
        else:
            i = i - 0.02
        # i = i - 0.001
    print("Minimal MSE: {}, Top Features: {}".format(min_values[0], len(min_values[1])))
    fig = plt.figure()
    p2 = fig.add_subplot(111)
    p2.plot(reduced_features[0], reduced_features[1], label='best features')
    if add_bad_feats:
        p2.plot(reduced_features[0], reduced_features_bad, label='worst features')
        p2.plot(_index_to_mse_drop_average)
    p2.set_ylabel('mean squared error of the predictions')
    p2.set_xlabel('size of the feature subset')
    p2.set_title('MSE of ridge regression models on feature subsets')
    p2.legend()
    plt.show()
    return min_values


def plot_4split_barcharts(_index_to_mse_drop_average):
    cutoff = 0.00001
    _mse_c1 = _index_to_mse_drop_average[:120]
    _mse_c2 = _index_to_mse_drop_average[120:1807]
    _mse_c3 = _index_to_mse_drop_average[1807:12471]
    _mse_chars = [x for x in _index_to_mse_drop_average[:12471] if abs(x) > cutoff]

    _mse_w1 = _index_to_mse_drop_average[12471:21318]
    _mse_w2 = _index_to_mse_drop_average[21318:33656]
    _mse_w3 = _index_to_mse_drop_average[33656:37332]
    _mse_words = [x for x in _index_to_mse_drop_average[12471:37332] if abs(x) > cutoff]

    _mse_engf = [x for x in _index_to_mse_drop_average[37332:37344] if abs(x) > cutoff]
    _mse_wl = [x for x in _index_to_mse_drop_average[37344:] if abs(x) > cutoff]

    fig = plt.figure()
    p2 = fig.add_subplot(411)
    p2.bar(np.arange(len(_mse_chars)), sorted(_mse_chars), width=1)
    p2.set_title("char n-grams")
    # p2.set_ylim(-0.003, 0.003, 0.004)
    # p2.minorticks_off()

    p5 = fig.add_subplot(412)
    # p5.plot(sorted(_mse_words))
    p5.bar(np.arange(len(_mse_words)), sorted(_mse_words), width=1)
    p5.set_title("word n-grams")
    # p5.set_ylim(-0.003, 0.003, 0.004)
    # p5.minorticks_off()
    # p5.set_yticklabels("")

    p8 = fig.add_subplot(413)
    p8.bar(np.arange(len(_mse_engf)), sorted(_mse_engf), width=1)
    p8.set_title("engineered features")
    # p8.set_ylim(-0.003, 0.003, 0.004)
    # p8.minorticks_off()
    # p8.set_yticklabels("")
    # p8.set_xticklabels("")
    p9 = fig.add_subplot(414)
    p9.bar(np.arange(len(_mse_wl)), sorted(_mse_wl), width=1)
    p9.set_title("wordlist features")
    # p9.set_ylim(-0.003, 0.003, 0.004)
    # p9.minorticks_off()
    # p9.set_yticklabels("")
    # p9.set_xticklabels("")
    # p2.plot(sorted(_index_to_mse_drop_max), label="maximum")
    # p2.plot(sorted(_index_to_mse_drop_min), label="minimum")
    # p2.plot(sorted(_index_to_mse_drop_average), label="average")
    # p2.set_ylabel('MSE Difference')
    # p2.set_xlabel('Feature Index')
    # p2.set_title('MSE Changes on Feature Removal')
    # p2.legend()
    plt.subplots_adjust(wspace=0.1, hspace=1)
    plt.show()


# get the training data
cbd = ClickbaitDataset("../../clickbait17-validation-170630/instances.jsonl",
                       "../../clickbait17-validation-170630/truth.jsonl")
cbm = ClickbaitModel()
ev_function = cbm.eval_regress
# f_builder = build_new_features(cbd)
f_builder = pickle.load(open("../feature_builder_w13_c13_reduced.pkl", "rb"))
x, x2, y, y2 = f_builder.build_features_split
feature_names = f_builder.feature_names


cbm.regress(x, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
y_predict = cbm.predict(x2)
# start_mse = skm.mean_squared_error(y2, y_predict)

idrop, imean, imin, imax = reconstruct_results()

_mean_indices = sorted(range(len(imean)), key=lambda x: imean[x])


cutoff = 0.00001
_mse_c1 = imean[:120]
_mse_c2 = imean[120:1807]
_mse_c3 = imean[1807:12471]
_mse_chars = imean[:12471]

_mse_w1 = imean[12471:21318]
_mse_w2 = imean[21318:33656]
_mse_w3 = imean[33656:37332]
_mse_words = imean[12471:37332]

_mse_engf = imean[37332:37344]
_mse_wl = imean[37344:]

_chars = deque()
_words = deque()
_engs = deque()
_wls = deque()
for index in range(len(imean)):
    if abs(imean[index]) > cutoff:
        if index >= 37344:
            _wls.append((imean[index], index))
        elif index >= 37332 and index < 37344:
            _engs.append((imean[index], index))
        elif index >= 12471 and index < 37332:
            _words.append((imean[index], index))
        elif index < 12471:
            _chars.append((imean[index], index))

with open("best_and_worst_features_by_category.txt", "w") as of:
    of.write("\ncharacter tokens: \n")
    for c in sorted(_chars, key=lambda x: x[0]):
        of.write("{}: {}\n".format(c[0], feature_names[c[1]]))

    of.write("\nword tokens: \n")
    for c in sorted(_words, key=lambda x: x[0]):
        of.write("{}: {}\n".format(c[0], feature_names[c[1]]))

    of.write("\nengineered features: \n")
    for c in sorted(_engs, key=lambda x: x[0]):
        of.write("{}: {}\n".format(c[0], feature_names[c[1]]))

    of.write("\nwordlists: \n")
    for c in sorted(_wls, key=lambda x: x[0]):
        of.write("{}: {}\n".format(c[0], feature_names[c[1]]))

print("draw")

plot_compare_techniques(x, x2, y, y2)

# min_values = plot_mse_reduced_sets(idrop, imean, x, x2, y, y2)

# train model on full dataset
'''
x = f_builder.build_features
x_train_subset = scipy.sparse.csc_matrix(x)[:, min_values[1]]
cbm.regress(x_train_subset, cbd.get_y(), Ridge(alpha=3.5, solver="sag"), evaluate=False)

cbd2 = ClickbaitDataset(instances_path="../../clickbait17-test-170720/instances.jsonl")
f_builder.build(cbd2)
x_test = f_builder.build_features
x_test_subset = scipy.sparse.csc_matrix(x_test)[:, min_values[1]]
y_predict = cbm.predict(x_test_subset)


id_list = sorted(cbd2.dataset_dict.keys())
_results_list = []
for i in range(len(id_list)):
    _results_list.append({'id': id_list[i], 'clickbaitScore': y_predict[i]})

with open("final_evaluation_predictions.jsonl", 'w') as of:
    for l in _results_list:
        of.write(json.dumps(l))
        of.write("\n")
'''
# start_mse = skm.mean_squared_error(y2, y_predict)

# use these indices to predict test dataset

# plot_4split_barcharts(imean)

# TODO find indices with high min/max spread -> candidates for interactions
# _pos = np.linspace(0, (len(_index_to_mse_drop_average)+1)/1000, len(_index_to_mse_drop_average))
'''
fig = plt.figure()
p2 = fig.add_subplot(331)
p2.plot(sorted(_mse_c1))
p2.set_title("char 1-grams")
p2.set_ylim(-0.008, 0.002, 0.004)
p2.minorticks_off()
# p2.set_yticklabels("")
# p2.set_xticklabels("")
p3 = fig.add_subplot(332)
p3.plot(sorted(_mse_c2))
p3.set_title("char 2-grams")
p3.set_ylim(-0.008, 0.002, 0.004)
p3.minorticks_off()
p3.set_yticklabels("")
# p3.set_xticklabels("")
p4 = fig.add_subplot(333)
p4.plot(sorted(_mse_c3))
p4.set_title("char 3-grams")
p4.set_ylim(-0.008, 0.002, 0.004)
p4.minorticks_off()
p4.set_yticklabels("")
# p4.set_xticklabels("")

p5 = fig.add_subplot(334)
p5.plot(sorted(_mse_w1))
p5.set_title("word 1-grams")
p5.set_ylim(-0.008, 0.002, 0.004)
p5.minorticks_off()
p5.set_yticklabels("")
# p5.set_xticklabels("")
p6 = fig.add_subplot(335)
p6.plot(sorted(_mse_w2))
p6.set_title("word 2-grams")
p6.set_ylim(-0.008, 0.002, 0.004)
p6.minorticks_off()
p6.set_yticklabels("")
# p6.set_xticklabels("")
p7 = fig.add_subplot(336)
p7.plot(sorted(_mse_w3))
p7.set_title("word 3-grams")
p7.set_ylim(-0.008, 0.002, 0.004)
p7.minorticks_off()
p7.set_yticklabels("")
# p7.set_xticklabels("")

p8 = fig.add_subplot(337)
p8.plot(sorted(_mse_engf))
p8.set_title("engineered features")
p8.set_ylim(-0.008, 0.002, 0.004)
p8.minorticks_off()
p8.set_yticklabels("")
# p8.set_xticklabels("")
p9 = fig.add_subplot(338)
p9.plot(sorted(_mse_wl))
p9.set_title("wordlist features")
p9.set_ylim(-0.008, 0.002, 0.004)
p9.minorticks_off()
p9.set_yticklabels("")
# p9.set_xticklabels("")'''



''' print list content
indices = _mean_indices[:100]
index_names = f_builder.feature_names
for i in indices:
    print(index_names[i])
'''


'''
print("word 1grama")
t = TfidfVectorizer(preprocessor=ft.preprocess, tokenizer=ft.tokenize, min_df=3, ngram_range=(1, 3))
print(t.fit_transform(cbd.get_x('postText')).shape)


print("char 1grama")
t = TfidfVectorizer(analyzer='char', preprocessor=ft.preprocess, tokenizer=ft.tokenize, min_df=3, ngram_range=(1, 3))
print(t.fit_transform(cbd.get_x('postText')).shape)
word 1grama
24861
char 1grama
12471


full set size
37528

char 1grama
120
char 2grama
1687
char 3grama
10664

word 1grama
8847
word 2grama
12338
word 3grama
3676

(hashtags_count, 'postText'),
(mentions_count, 'postText'),
(sentiment_polarity, 'postText'),
(flesch_kincait_score, 'postText'),
(has_abbrev, 'postText'),
(number_of_dots, 'postText'),
(start_with_number, 'postText'),
(longest_word_length, 'postText'),
(mean_word_length, 'postText'),
(char_sum, 'postText'),
(has_media_attached, 'postMedia'),
(part_of_day, 'postTimestamp'),
12 engineered

(easy_words_ratio, 'postText'),
(stop_word_ratio, 'postText'),
(clickbait_phrases_count, 'postText'))
181 gi lists
= 184 text lists
'''
