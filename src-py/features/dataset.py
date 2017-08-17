#!/usr/bin/python3
import json
import numpy as np


class ClickbaitDataset(object):
    # TODO switch to pandas
    def __init__(self, instances_path=None, truth_path=None):
        self.dataset_dict = {}
        if instances_path is not None and truth_path is not None:
            self.from_file(instances_path, truth_path)

    def from_file(self, instances_path, truth_path):
        with open(instances_path, "r") as inf:
            _instances = [json.loads(x) for x in inf.readlines()]
        with open(truth_path, "r") as inf:
            _truth = [json.loads(x) for x in inf.readlines()]

        for i in _instances:
            self.dataset_dict[i['id']] = {'postTimestamp': i['postTimestamp'],
                                          'postText': i['postText'],
                                          'postMedia': i['postMedia'],
                                          'targetTitle': i['targetTitle'],
                                          'targetDescription': i['targetDescription'],
                                          'targetKeywords': i['targetKeywords'],
                                          'targetParagraphs': i['targetParagraphs'],
                                          'targetCaptions': i['targetCaptions']}

        for t in _truth:
            self.dataset_dict[t['id']]['truthMean'] = t['truthMean']
            self.dataset_dict[t['id']]['truthClass'] = t['truthClass']

        # self.id_index = {index: key for index, key in enumerate(self.dataset_dict.keys())}

    def add_tweet(self, tweet_id, post_timestamp='', post_text=[], post_media=[], target_title='', target_description='',
                  target_keywords='', target_paragraphs=[], target_captions=[]):
        self.dataset_dict[tweet_id] = {'postTimestamp': post_timestamp,
                                       'postText': post_text,
                                       'postMedia': post_media,
                                       'targetTitle': target_title,
                                       'targetDescription': target_description,
                                       'targetKeywords': target_keywords,
                                       'targetParagraphs': target_paragraphs,
                                       'targetCaptions': target_captions,
                                       'truthMean': None,
                                       'truthClass': None}
        return self

    def get_y(self):
        # return np.asarray([self.dataset_dict[self.id_index[key]]['truthMean'] for key in sorted(self.id_index.keys())])
        return np.asarray([self.dataset_dict[key]['truthMean'] for key in sorted(self.dataset_dict.keys())])

    def get_y_class(self):
        class_list = [self.dataset_dict[key]['truthClass'] for key in sorted(self.dataset_dict.keys())]
        return np.asarray([0 if t == "no-clickbait" else 1 for t in class_list])

    def get_x(self, field_name):
        # TODO dont just use the first element in the text
        return np.asarray([self.dataset_dict[key][field_name][0]
                           for key in sorted(self.dataset_dict.keys())])

    def size(self):
        return len(self.dataset_dict.keys())


if __name__ == "__main__":
    pass
