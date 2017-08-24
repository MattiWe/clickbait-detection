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

# get list of scores and a list of the postTexts
cbd = ClickbaitDataset("../../clickbait17-validation-170630/instances.jsonl", "../../clickbait17-validation-170630/truth.jsonl")
common_phrases = ft.ContainsWordsFeature("../wordlists/TerrierStopWordList.txt", ratio=True)
char_3grams = ft.NGramFeature(TfidfVectorizer, o=3, analyzer='char', fit_data=cbd.get_x('postText'))
word_3grams = ft.NGramFeature(TfidfVectorizer, o=3, fit_data=cbd.get_x('postText'))
# stop_word_count = ContainsWordsFeature(data, wordlist, only_words=True, ratio=False)

stop_word_ratio = ft.ContainsWordsFeature("../wordlists/TerrierStopWordList.txt", ratio=True)
easy_words_ratio = ft.ContainsWordsFeature("../wordlists/DaleChallEasyWordList.txt", ratio=True)
mentions_count = ft.ContainsWordsFeature(['@'], only_words=False, ratio=False)
hashtags_count = ft.ContainsWordsFeature(['#'], only_words=False, ratio=False)
clickbait_phrases_count = ft.ContainsWordsFeature("../wordlists/DownworthyCommonClickbaitPhrases.txt",
                                                  only_words=False, ratio=False)

f_builder = FeatureBuilder((char_3grams, 'postText'),
                           (word_3grams, 'postText'),
                           (stop_word_ratio, 'postText'),
                           (easy_words_ratio, 'postText'),
                           (mentions_count, 'postText'),
                           (hashtags_count, 'postText'),
                           (clickbait_phrases_count, 'postText'))
x_train, x_test, y, y2 = f_builder.build(cbd, split=True)

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
    for i in range(1000):
        of.write(json.dumps({"selectedFeatures": [1]*int(x_train.shape[1]), "runs": []}))
        of.write("\n")
