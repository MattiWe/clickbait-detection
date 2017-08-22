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

# print(sys.argv)

# get list of scores and a list of the postTexts
cbd = ClickbaitDataset("../clickbait17-validation-170630/instances.jsonl", "../clickbait17-validation-170630/truth.jsonl")
common_phrases = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
char_3grams = ft.NGramFeature(TfidfVectorizer, o=3, analyzer='char', fit_data=cbd.get_x('postText'))
word_3grams = ft.NGramFeature(TfidfVectorizer, o=3, fit_data=cbd.get_x('postText'))
# stop_word_count = ContainsWordsFeature(data, wordlist, only_words=True, ratio=False)

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
y = cbd.get_y()
# save for hadoop task
# scipy.sparse.save_npz("x_train", x)
# np.savez("y_train", data=y.data, shape=y.shape)
# scipy.sparse.save_npz("x_test", x2)
# np.savez("y_test", data=y2.data, shape=y2.shape)

# y = np.asarray([0 if t < 0.5 else 1 for t in y])
# y2 = np.asarray([0 if t < 0.5 else 1 for t in y2])

# Test classification
cbm = ClickbaitModel()
cbm.regress(x, y, "RandomForestRegressor", evaluate=False)

# save needed data
c3g = char_3grams.vectorizer_fit
w3g = word_3grams.vectorizer_fit
pickle.dump(c3g, open("c3g.pkl", 'wb'))
pickle.dump(w3g, open("w3g.pkl", 'wb'))
cbm.save("cbm_rfr.pkl")
'''
# metrics
ev_function = cbm.eval_classify
print("\n real predictions")
y_predict = cbm.predict(x2)
ev_function(y2, y_predict)

print("\n shuffeled")
y_predict = deepcopy(y2)
np.random.shuffle(y_predict)
ev_function(y2, y_predict)
'''

'''_id = 608310377143799810
_txt = ["U.S. Soccer should start answering tough questions about Hope Solo, @eric_adelson writes."]

prediction = cbm.predict(f_builder.build(ClickbaitDataset().add_tweet(tweet_id=_id, post_text=_txt)))
print(prediction)
'''
