#!/usr/bin/python3
from sklearn import svm
import sklearn.metrics as skm
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split


def normalized_mean_squared_error(truth, predictions):
    norm = skm.mean_squared_error(truth, np.full(len(truth), np.mean(truth)))
    return skm.mean_squared_error(truth, predictions) / norm


class ClickbaitModel(object):
    __regression_measures = {'Explained variance': skm.explained_variance_score,
                             'Mean absolute error': skm.mean_absolute_error,
                             'Mean squared error': skm.mean_squared_error,
                             'Median absolute error': skm.median_absolute_error,
                             'R2 score': skm.r2_score,
                             'Normalized mean squared error': normalized_mean_squared_error}

    __classification_measures = {'Accuracy': skm.accuracy_score,
                                 'Precision': skm.precision_score,
                                 'Recall': skm.recall_score,
                                 'F1 score': skm.f1_score}

    def __init__(self):
        self.models = {"LogisticRegression": LogisticRegression(),
                       "MultinomialNB": MultinomialNB(),
                       "RandomForestClassifier": RandomForestClassifier(),
                       "SVR": svm.SVR(),
                       "RandomForestRegressor": RandomForestRegressor()}
        self.model_trained = None

    def classify(self, x, y, model, evaluate=True):
        if isinstance(model, str):
            self.model_trained = self.models[model]
        else:
            self.model_trained = model
        if evaluate:
            x_train, x_test, y_train, y_test = train_test_split(x, y.T, random_state=42)
        else:
            x_train = x
            y_train = y

        self.model_trained.fit(x_train, y_train)

        if evaluate:
            y_predicted = self.model_trained.predict(x_train)
            y_scores = self.model_trained.predict_proba(x_test)
            for cm in self.__classification_measures:
                print("{}: {}".format(cm, self.__classification_measures[cm](y_train, y_predicted)))
            print("ROC-AUC: {}".format(skm.roc_auc_score(y_train, y_predicted)))

    def regress(self, features, model, evaluate=True):
        if isinstance(model, str):
            self.model_trained = self.models[model]
        else:
            self.model_trained = model
        if evaluate:
            x_train, x_test, y_train, y_test = train_test_split(features, self.data.get_y_class().T, random_state=42)
        else:
            x_train = features
            y_train = data.get_y_class()

        self.model_trained.fit(x_train, y_train)

        if evaluate:
            y_predicted = self.model_trained.predict(x_test)
            for rm in __regression_measures:
                print("{}: {}".format(rm, self.__regression_measures[rm](y_test, y_predicted)))

    def predict(self, x):
        return self.model_trained.predict(x)

    def eval_classify(self, y_test, y_predicted):
        for cm in self.__classification_measures:
            print("{}: {}".format(cm, self.__classification_measures[cm](y_test, y_predicted)))

    def eval_regress(self, y_test, y_predicted):
        for rm in __regression_measures:
            print("{}: {}".format(rm, self.__regression_measures[rm](y_test, y_predicted)))
