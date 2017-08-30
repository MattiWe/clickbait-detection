#!/usr/bin/python3
import numpy as np
from features import feature as ft
from features.feature_builder import FeatureBuilder
from features.ml import ClickbaitModel
from features.dataset import ClickbaitDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from copy import deepcopy
import scipy.sparse
import sys
import json
import pickle
import os


def build_new_features(cbd):
    # get list of scores and a list of the postTexts
    common_phrases = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
    char_3grams = ft.NGramFeature(TfidfVectorizer, o=3, analyzer='char', fit_data=cbd.get_x('postText'))
    word_3grams = ft.NGramFeature(TfidfVectorizer, o=3, fit_data=cbd.get_x('postText'))
    # stop_word_count = ContainsWordsFeature(data, wordlist, only_words=True, ratio=False)

    stop_word_ratio = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
    easy_words_ratio = ft.ContainsWordsFeature("wordlists/DaleChallEasyWordList.txt", ratio=True)
    mentions_count = ft.ContainsWordsFeature(['@'], only_words=False)
    hashtags_count = ft.ContainsWordsFeature(['#'], only_words=False)
    clickbait_phrases_count = ft.ContainsWordsFeature("wordlists/DownworthyCommonClickbaitPhrases.txt",
                                                      only_words=False)
    flesch_kincait_score = ft.FleschKincaidScore()
    has_abbrev = ft.ContainsWordsFeature("wordlists/OxfortAbbreviationsList.txt", only_words=False, binary=True)
    number_of_dots = ft.ContainsWordsFeature(['.'], only_words=False)
    start_with_number = ft.StartsWithNumber()
    longest_word_length = ft.LongestWordLength()
    mean_word_length = ft.MeanWordLength()
    char_sum = ft.CharacterSum()
    has_media_attached = ft.HasMediaAttached()
    part_of_day = ft.PartOfDay()

    # TODO content features
    # TODO sentiment polarity
    f_builder = FeatureBuilder((word_3grams, 'postText'),  # (char_3grams, 'postText'),
                               (hashtags_count, 'postText'),
                               (mentions_count, 'postText'),
                               (flesch_kincait_score, 'postText'),
                               (stop_word_ratio, 'postText'),
                               (easy_words_ratio, 'postText'),
                               (has_abbrev, 'postText'),
                               (number_of_dots, 'postText'),
                               (start_with_number, 'postText'),
                               (longest_word_length, 'postText'),
                               (mean_word_length, 'postText'),
                               (char_sum, 'postText'),
                               (has_media_attached, 'postMedia'),
                               (part_of_day, 'postTimestamp'),
                               (clickbait_phrases_count, 'postText'))

    for file_name in os.listdir("wordlists/general-inquirer"):
        f = ft.ContainsWordsFeature("wordlists/general-inquirer/" + file_name)
        f_builder.add_feature(feature=f, data_field_name='postText')

    print('building')
    f_builder.build(cbd, split=True)
    pickle.dump(obj=f_builder, file=open("feature_builder_w13_c00_2.pkl", "wb"))
    return f_builder


cbd = ClickbaitDataset("../clickbait17-validation-170630/instances.jsonl", "../clickbait17-validation-170630/truth.jsonl")
f_builder = build_new_features(cbd)
# f_builder = pickle.load(open("feature_builder.pkl", "rb"))
x, x2, y, y2 = f_builder.build_features
# x = f_builder.build(cbd)
# y = cbd.get_y()
# save for hadoop task
# scipy.sparse.save_npz("x_train", x)
# np.savez("y_train", data=y.data, shape=y.shape)
# scipy.sparse.save_npz("x_test", x2)
# np.savez("y_test", data=y2.data, shape=y2.shape)
# save needed data
'''c3g = char_3grams.vectorizer_fit
w3g = word_3grams.vectorizer_fit
pickle.dump(c3g, open("c3g.pkl", 'wb'))
pickle.dump(w3g, open("w3g.pkl", 'wb'))
cbm.save("cbm_rfr.pkl")
'''
# y = np.asarray([0 if t < 0.5 else 1 for t in y])
# y2 = np.asarray([0 if t < 0.5 else 1 for t in y2])

# Test classification

print('training')
'''cbm = ClickbaitModel()
ev_function = cbm.eval_regress
cbm.regress(x, y, Ridge(alpha=3.5, solver="sag"), evaluate=False)
# cbm.save("model_trained.pkl")
# cbm.load("model_trained.pkl")
y_predict = cbm.predict(x2)
ev_function(y2, y_predict)'''

'''_id = 608310377143799810
_txt = ["U.S. Soccer should start answering tough questions about Hope Solo, @eric_adelson writes."]

prediction = cbm.predict(f_builder.build(ClickbaitDataset().add_tweet(tweet_id=_id, post_text=_txt)))
print(prediction)
'''
