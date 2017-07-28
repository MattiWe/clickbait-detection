#!/usr/bin/python3
import numpy as np
import json


def parse_input(instances_path, truth_path):
    instances = []
    truth = []

    with open(instances_file_train, "r") as inf:
        instances = [json.loads(x) for x in inf.readlines()]
    with open(truth_file_train, "r") as inf:
        truth = [json.loads(x) for x in inf.readlines()]

    return instances, truth


def build_dataset(instances, truth):
    # TODO switch to pandas
    dataset = {}
    for i in instances:
        dataset[i['id']] = {'postText': i['postText'], 'targetTitle': i['targetTitle'],
                            'targetDescription': i['targetDescription'], 'targetKeywords': i['targetKeywords'],
                            'targetParagraphs': i['targetParagraphs'], 'targetCaptions': i['targetCaptions']}
    for t in truth:
        dataset[t['id']]['truthMean'] = t['truthMean']

    return dataset


class Feature(object):

    def __init__(self, name):
        self.name = ""
        pass

    def asarray(self):
        pass


class SparseFeature(Feature):

    def __init__(self, name, vectorizer, data, n=None):
        self.name = name
        self.vectorizer = vectorizer


class SingleVectorFeature(feature):

    def __init__(self):
        pass


class FeatureBuilder(object):

    def __init__(self):
        pass


if __name__ == "__main__":
    pass
