#!/usr/bin/python3
import numpy as np
# import json
# from itertools import combinations
from collections import deque
# import random
from math import ceil

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer, sent_tokenize
from nltk.corpus import cmudict
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import string
from sklearn.feature_extraction.text import TfidfVectorizer

from scipy.sparse import csc_matrix

lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)


def tokenize(text):
    global tknzr
    return [token for token in tknzr.tokenize(text)]  # if token not in string.punctuation]


def word_ngram_tokenize(text):
    global tknzr
    return [token for token in tknzr.tokenize(text) if token not in [',', '.', '!', '?']]


def pos_tokenize(text):
    global tknzr
    punctuation = string.punctuation
    token_list = [[token] for token in tknzr.tokenize(text) if token not in string.punctuation]
    return token_list


def preprocess(string):
    global lemmatizer
    string = string.lower()
    if string.endswith("'s"):
        string = string[:-2]
    string = lemmatizer.lemmatize(string)
    return string


def preprocess_tokenize(text):
    for token in tokenize(text):
        token = preprocess(token)
        yield token


class Feature(object):

    def __init__(self, feature=None):
        self.feature = feature
        self.name = [type(self)]

    def aslist(self):
        return self.feature

    def assparse(self):
        return csc_matrix(self.feature)


class NGramFeature(Feature):
    def __init__(self, vectorizer, analyzer='word', n=1, o=1, cutoff=1, fit_data=None, vocab=None):
        self.vectorizer = vectorizer(analyzer=analyzer, preprocessor=preprocess,
                                     tokenizer=tokenize, min_df=cutoff, ngram_range=(n, o), vocabulary=vocab)
        if fit_data is not None:
            self.fit(fit_data)

    def get_vocab(self):
        return self.vectorizer_fit.vocabulary_

    def fit(self, data):
        self.vectorizer_fit = self.vectorizer.fit(data)
        self.name = [None] * len(self.vectorizer_fit.vocabulary_)
        for key, value in self.vectorizer_fit.vocabulary_.items():
            self.name[value] = key

    def assparse(self, data):
        return csc_matrix(self.vectorizer_fit.transform(data))


class ContainsWordsFeature(Feature):
    def __init__(self, wordlist, only_words=True, ratio=False, binary=False):
        self.wordlist = wordlist
        if isinstance(wordlist, str):
            with open(wordlist, "r") as inf:
                self.wordlist = [x.strip() for x in inf.readlines()]
        self.only_words = only_words
        self.ratio = ratio
        self.binary = binary
        self.name = [str(wordlist)]

    def assparse(self, data):
        _result = deque()
        for tweet in data:
            _processed = [preprocess(x) for x in tokenize(tweet)]
            _processed_string = ''.join(_processed)
            _tmp = 0
            for word in self.wordlist:
                if self.only_words:
                    _tmp += _processed.count(word)
                elif not self.only_words:
                    _tmp += _processed_string.count(word)
            try:
                _result.append(_tmp if not self.ratio else _tmp / len(_processed))
            except ZeroDivisionError:
                _result.append(0)
        if self.binary:
            _result = [0 if i == 0 else 1 for i in _result]

        return np.asarray(list(_result))[:, np.newaxis]


class FleschKincaidScore(Feature):

    def __init__(self):
        self.prondict = cmudict.dict()
        self.name = ["Flesch-Kincaid Score"]

    def assparse(self, data):
        def get_pron(word):
            try:
                return self.prondict[word][0]
            except KeyError:
                return [['1']]
        _result = []
        for tweet in data:
            _processed = [preprocess(x) for x in tokenize(tweet)]
            word_count = len(_processed)
            sent_count = len(sent_tokenize(tweet))
            syllable_count = np.sum([len([s for s in get_pron(word)
                                          if (s[-1]).isdigit()])
                                     for word in _processed])
            try:
                _result.append(0.39 *
                               (word_count / sent_count) + 11.8 *
                               (syllable_count / word_count) - 15.59)
            except ZeroDivisionError:
                _result.append(-3.4)  # thats the minimal possible FK-Score

        return np.asarray(list(_result))[:, np.newaxis]


class StartsWithNumber(Feature):
    def assparse(self, data):
        _result = []
        for tweet in data:
            _processed = [preprocess(x) for x in tokenize(tweet)]
            try:
                _result.append(1 if _processed[0].isdigit() else 0)
            except IndexError:
                _result.append(0)
        return np.asarray(list(_result))[:, np.newaxis]


class LongestWordLength(Feature):
    def assparse(self, data):
        _result = []
        for tweet in data:
            _processed_len = [len(preprocess(x)) for x in tokenize(tweet)]
            try:
                _result.append(max(_processed_len))
            except ValueError:
                _result.append(0)
        return np.asarray(list(_result))[:, np.newaxis]


class MeanWordLength(Feature):
    def assparse(self, data):
        _result = []
        for tweet in data:
            _processed_len = [len(preprocess(x)) for x in tokenize(tweet)]
            try:
                _result.append(sum(_processed_len) / len(_processed_len))
            except ZeroDivisionError:
                _result.append(0)
        return np.asarray(list(_result))[:, np.newaxis]


class CharacterSum(Feature):
    def assparse(self, data):
        _result = []
        for tweet in data:
            _processed = [preprocess(x) for x in tokenize(tweet)]
            try:
                _result.append(sum(len(x) for x in _processed))
            except ZeroDivisionError:
                _result.append(0)
        return np.asarray(list(_result))[:, np.newaxis]


class HasMediaAttached(Feature):
    def assparse(self, data):
        _result = []
        for tweet in data:
            _result.append(1 if tweet else 0)
        return np.asarray(list(_result))[:, np.newaxis]


class PartOfDay(Feature):
    def assparse(self, data):
        _result = []
        for tweet in data:
            d = tweet.split(':')[0]
            _result.append(ceil(int(d.split()[3]) / 6))
        return np.asarray(list(_result))[:, np.newaxis]


class SentimentPolarity(Feature):
    def __init__(self):
        self.sid = SentimentIntensityAnalyzer()
        self.name = ["Sentiment Polarity"]

    def assparse(self, data):
        _result = []
        for tweet in data:
            _processed = tokenize(tweet)
            try:
                _result.append(sid.polarity_scores(" ".join(_processed))["compound"])
            except Exception:
                _result.append(0)


if __name__ == "__main__":
    pass
