#!/usr/bin/python3
import numpy as np
from features import feature as ft
from features.feature_builder import FeatureBuilder
from features.ml import ClickbaitModel
from features.dataset import ClickbaitDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from copy import deepcopy
import scipy.sparse
import sys
import json
import pickle

c3g_vocab = pickle.load(open("c3g.pkl", "rb"))
w3g_vocab = pickle.load(open("w3g.pkl", "rb"))

cbd = ClickbaitDataset(instances_path=sys.argv[1])
common_phrases = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
char_3grams = ft.NGramFeature(TfidfVectorizer, o=3, analyzer='char', fit_data=cbd.get_x('postText'))
char_3grams.vectorizer_fit = c3g_vocab
word_3grams = ft.NGramFeature(TfidfVectorizer, o=3, fit_data=cbd.get_x('postText'))
word_3grams.vectorizer_fit = w3g_vocab

stop_word_ratio = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
easy_words_ratio = ft.ContainsWordsFeature("wordlists/DaleChallEasyWordList.txt", ratio=True)
mentions_count = ft.ContainsWordsFeature(['@'], only_words=False, ratio=False)
hashtags_count = ft.ContainsWordsFeature(['#'], only_words=False, ratio=False)
clickbait_phrases_count = ft.ContainsWordsFeature("wordlists/DownworthyCommonClickbaitPhrases.txt",
                                                  only_words=False, ratio=False)

f_builder = FeatureBuilder((char_3grams, 'postText'),
                           (word_3grams, 'postText'),
                           (stop_word_ratio, 'postText'),
                           (easy_words_ratio, 'postText'),
                           (mentions_count, 'postText'),
                           (hashtags_count, 'postText'),
                           (clickbait_phrases_count, 'postText'))
# x, x2, y, y2 = f_builder.build(cbd, split=True)
x = f_builder.build(cbd)


cbm = ClickbaitModel()
cbm.load("cbm_rfr.pkl")

y = cbm.predict(x)

# TODO print x to file
id_list = sorted(cbd.dataset_dict.keys())
_results_list = []
for i in range(len(id_list)):
    _results_list.append({'id': id_list[i], 'clickbaitScore': y[i]})

with open(sys.argv[2], 'w') as of:
    for l in _results_list:
        of.write(json.dumps(l))
        of.write("\n")
