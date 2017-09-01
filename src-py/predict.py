#!/usr/bin/python3
from features.feature_builder import FeatureBuilder
from features.ml import ClickbaitModel
from features.dataset import ClickbaitDataset
import sys
import json
import pickle

cbd = ClickbaitDataset(instances_path=sys.argv[1])
f_builder = pickle.load(open("feature_builder_complete.pkl", "rb"))
f_builder.build(cbd)
x = f_builder.build_features

cbm = ClickbaitModel()
cbm.load("model_trained.pkl")

y = cbm.predict(x)

id_list = sorted(cbd.dataset_dict.keys())
_results_list = []
for i in range(len(id_list)):
    _results_list.append({'id': id_list[i], 'clickbaitScore': y[i]})

with open(sys.argv[2], 'w') as of:
    for l in _results_list:
        of.write(json.dumps(l))
        of.write("\n")
