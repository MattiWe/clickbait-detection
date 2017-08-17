#!/usr/bin/python3
import numpy as np
# import json
# from itertools import combinations
from collections import deque
# import random

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
import string

from scipy.sparse import csc_matrix

lemmatizer = WordNetLemmatizer()
tknzr = TweetTokenizer(strip_handles=True, reduce_len=True)


def tokenize(text):
    global tknzr
    return [token for token in tknzr.tokenize(text)]  # if token not in string.punctuation]


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
    def __init__(self, featue):
        self.feature = feature

    def aslist(self):
        return self.feature

    def assparse(self):
        return csc_matrix(self.feature)


class NGramFeature(Feature):
    # TODO get() method, so only compute when building a matrix with this feature
    def __init__(self, vectorizer, analyzer='word', n=None, o=None, fit_data=None):
        if n is None:
            n, o = 1, 1
        elif o is None:
            o = n
        self.vectorizer = vectorizer(preprocessor=preprocess, tokenizer=tokenize, ngram_range=(n, o))
        if fit_data is not None:
            self.vectorizer_fit = self.vectorizer.fit(fit_data)

    def fit(self, data):
        self.vectorizer_fit = self.vectorizer.fit(data)

    def assparse(self, data):
        return csc_matrix(self.vectorizer_fit.transform(data))


class ContainsWordsFeature(Feature):

    def __init__(self, wordlist, only_words=True, ratio=False):
        self.wordlist = wordlist
        if isinstance(wordlist, str):
            with open(wordlist, "r") as inf:
                self.wordlist = [x.strip() for x in inf.readlines()]
        self.only_words = only_words
        self.ratio = ratio

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

        return np.asarray(list(_result))[:, np.newaxis]


if __name__ == "__main__":
    pass
