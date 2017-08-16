#!/usr/bin/python3
from features import feature as ft
from features.feature_builder import FeatureBuilder
from features.ml import ClickbaitModel
from features.dataset import ClickbaitDataset
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer


# get list of scores and a list of the postTexts
cbd = ClickbaitDataset("../clickbait17-validation-170630/instances.jsonl", "../clickbait17-validation-170630/truth.jsonl")
common_phrases = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
char_3grams = ft.NGramFeature(TfidfVectorizer, o=3, analyzer='char')
word_3grams = ft.NGramFeature(TfidfVectorizer, o=3)
# stop_word_count = ContainsWordsFeature(data, wordlist, only_words=True, ratio=False)

stop_word_ratio = ft.ContainsWordsFeature("wordlists/TerrierStopWordList.txt", ratio=True)
easy_words_ratio = ft.ContainsWordsFeature("wordlists/DaleChallEasyWordList.txt", ratio=True)
mentions_count = ft.ContainsWordsFeature(['@'], only_words=False, ratio=False)
hashtags_count = ft.ContainsWordsFeature(['#'], only_words=False, ratio=False)
clickbait_phrases_count = ft.ContainsWordsFeature("wordlists/DownworthyCommonClickbaitPhrases.txt",
                                                  only_words=False, ratio=False)

f_builder = FeatureBuilder((char_3grams, 'postText'),
                           (word_3grams, 'postText'),
                           (stop_word_ratio, 'postText'),
                           (easy_words_ratio, 'postText'),
                           (mentions_count, 'postText'),
                           (hashtags_count, 'postText'),
                           (clickbait_phrases_count, 'postText'))
x = f_builder.build(cbd)

# Test classification
cbm = ClickbaitModel()
cbm.classify(x, cbd.get_y_class(), "RandomForestClassifier")
