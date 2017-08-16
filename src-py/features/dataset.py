#!/usr/bin/python3
import json
import numpy as np


class ClickbaitDataset(object):
    # TODO switch to pandas
    def __init__(self, instances_path, truth_path):
        with open(instances_path, "r") as inf:
            _instances = [json.loads(x) for x in inf.readlines()]
        with open(truth_path, "r") as inf:
            _truth = [json.loads(x) for x in inf.readlines()]

        self.dataset_dict = {}
        for i in _instances:
            self.dataset_dict[i['id']] = {'postText': i['postText'],
                                          'targetTitle': i['targetTitle'],
                                          'targetDescription': i['targetDescription'],
                                          'targetKeywords': i['targetKeywords'],
                                          'targetParagraphs': i['targetParagraphs'],
                                          'targetCaptions': i['targetCaptions']}

        for t in _truth:
            self.dataset_dict[t['id']]['truthMean'] = t['truthMean']
            self.dataset_dict[t['id']]['truthClass'] = t['truthClass']

        self.id_index = {index: key for index, key in enumerate(self.dataset_dict.keys())}

    def get_y(self):
        return np.asarray([self.dataset_dict[self.id_index[key]]['truthMean'] for key in sorted(self.id_index.keys())])

    def get_y_class(self):
        class_list = [self.dataset_dict[self.id_index[key]]['truthClass'] for key in sorted(self.id_index.keys())]
        return np.asarray([1 if t == "no-clickbait" else 0 for t in class_list])

    def get_x(self, field_name):
        # TODO dont just use the first element in the text
        return np.asarray([self.dataset_dict[self.id_index[key]][field_name][0]
                           for key in sorted(self.id_index.keys())])

    def size(self):
        return len(self.dataset_dict.keys())
