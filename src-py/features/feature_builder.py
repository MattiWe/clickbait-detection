#!/usr/bin/python3
from scipy.sparse import hstack
from features.feature import Feature


class FeatureBuilder(object):

    def __init__(self, *args):
        self.features = list(args)

    def add_feature(self, feature, data_field_name):
        self.features.append((feature, data_field_name))
        return self

    def build(self, data, split=False):
        _result = None

        def push(result, f):
            if result is None:
                result = f
            else:
                result = hstack((_result, f))
            return result

        for f in self.features:
            if isinstance(f[0], Feature):
                _result = push(_result, f[0].assparse(data.get_x(f[1])))

        if split:
            return train_test_split(_result, np.asarray(data.get_y()).T, random_state=42)
        return _result

    ''' TODO evaluate usefullness
    def compose(self, *args):
        return self.build(_features=[self.features[f] for f in args])

    def compose_split(self, *args):
        _feat, _val = self.build(_features=[self.features[f] for f in args])
        return train_test_split(_feat, np.asarray(_val).T, random_state=42)
    '''
