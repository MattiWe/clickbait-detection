#!/usr/bin/python3
import numpy as np
import json
from itertools import combinations

from nltk.stem.wordnet import WordNetLemmatizer
from nltk import word_tokenize
from nltk.tokenize import TweetTokenizer
import string

import scipy.sparse
from sklearn import svm
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer

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
    string = lemmatizer.lemmatize(string)
    if string.endswith("'s"):
        string = string[:-2]
    return string


def preprocess_tokenize(text):
    for token in tokenize(text):
        token = preprocess(token)
        yield token


class ClickbaitDataset(object):
    # TODO switch to pandas
    def __init__(self, instances_path, truth_path):
        with open(instances_path, "r") as inf:
            _instances = [json.loads(x) for x in inf.readlines()]
        with open(truth_path, "r") as inf:
            _truth = [json.loads(x) for x in inf.readlines()]

        self.dataset_dict = {}
        for i in _instances:
            self.dataset_dict[i['id']] = {'postText': i['postText'], 'targetTitle': i['targetTitle'],
                                          'targetDescription': i['targetDescription'], 'targetKeywords': i['targetKeywords'],
                                          'targetParagraphs': i['targetParagraphs'], 'targetCaptions': i['targetCaptions']}
        for t in _truth:
            self.dataset_dict[t['id']]['truthMean'] = t['truthMean']

        self.id_index = {index: key for index, key in enumerate(self.dataset_dict.keys())}

    def get_y(self):
        return [self.dataset_dict[self.id_index[key]]['truthMean'] for key in sorted(self.id_index.keys())]

    # TODO generic for all columns
    def get_x_posttext(self):
        # TODO dont just use the first element in the text
        return np.asarray([self.dataset_dict[self.id_index[key]]['postText'][0]
                           for key in sorted(self.id_index.keys())])

    def size(self):
        return len(self.dataset_dict.keys())


class Feature(object):
    def __init__(self, data):
        self.data = data

    def aslist(self):
        return []

    def assparse(self):
        return scipy.sparse.csc_matrix((self.data.size(), 1))


class NGramFeature(Feature):
    def __init__(self, data, vectorizer, n=None, o=None):
        self.data = data
        if n is None:
            n, o = 1, 1
        elif o is None:
            o = n
        self.vectorizer = vectorizer(preprocessor=preprocess, tokenizer=tokenize, ngram_range=(n, o))
        self.feature = self.vectorizer.fit_transform(self.data.get_x_posttext())

    def assparse(self):
        return self.feature


class SingleVectorFeature(Feature):

    def __init__(self):
        # cb_feat_wordlength[:, np.newaxis]
        pass


class FeatureBuilder(object):

    def __init__(self, data):
        self.data = data
        self.features = {}

    def add_feature(self, name, feature):
        self.features[name] = feature
        return self

    def build(self, _features=None, split=False):
        _result = scipy.sparse.csc_matrix((self.data.size(), 1))
        __features = self.features.values() if _features is None else _features
        for f in __features:
            if isinstance(f, Feature):
                _result = scipy.sparse.hstack((_result, f.assparse()))
            elif isinstance(f, scipy.sparse.spmatrix):
                _result = scipy.sparse.hstack((_result, f))
            elif isinstance(f, np.array):
                _result = scipy.sparse.hstack((_result, f[:, np.newaxis]))
            elif isinstance(f, np.ndarray):
                _result = scipy.sparse.hstack((_result, scipy.sparse.csc_matrix(f)))

        if split:
            return train_test_split(_result, np.asarray(self.data.get_y()).T, random_state=42)
        return _result, self.data.get_y()

    def compose(self, *args):
        return self.build(_features=[self.features[f] for f in args])

    def compose_split(self, *args):
        _feat, _val = self.build(_features=[self.features[f] for f in args])
        return train_test_split(_feat, np.asarray(_val).T, random_state=42)

    def get_combinations(self, m=1, n=1):
        '''Returns every n-o column combination of the build features as a list.
            E.g. for 1-2: every column and every 2-combination of columns'''
        list_of_features = [scipy.sparse.csc_matrix(x_train[:, x]) for x in range(x_train.shape[1])]
        _result = []
        for i in range(m, n + 1):
            comb = combinations(list_of_features, i)
            for j in comb:
                x = j[0]
                if len(j > 1):
                    for k in range(1, len(j)):
                        j = scipy.sparse.hstack((x, j[k]))
                _result.append(x)
        return _result


if __name__ == "__main__":
    # get list of scores and a list of the postTexts
    cbd = ClickbaitDataset("../clickbait17-train-170331/instances.jsonl", "../clickbait17-train-170331/truth.jsonl")
    f = NGramFeature(data=cbd, vectorizer=CountVectorizer, n=1)

    x_train, x_test, y_train, y_test = FeatureBuilder(cbd).add_feature("charonegramcount", f) \
                                                          .add_feature('test', f.assparse())  \
                                                          .compose_split('test')

    print(x_train.shape)
    model = svm.SVR()
    model.fit(x_train, y_train)
    y_predicted = model.predict(x_test)
    print(mean_squared_error(y_test, y_predicted))
