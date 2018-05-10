# -*- coding:utf-8 -*-
# !/usr/bin/env python

import pickle
import os
import jieba
import jieba.analyse
import random
import re
import statistics

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from collections import Counter
from gensim.models.word2vec import Word2Vec


def count_information():
    png_path = os.path.join(os.path.dirname(__file__), 'data/png/length.png')
    df = pd.read_csv(os.path.join(os.path.dirname(__file__), 'data/csv/train.csv')).dropna()

    context, length = df["comment_text"].values, []
    for index in context:
        length.append(len(index.split()))

    length_mean = statistics.mean(length)
    length_std = statistics.stdev(length)

    Dict = sorted(Counter(length).items(), key=lambda val: val[0], reverse=False)
    key, value = zip(*Dict)
    plt.plot(key, value)
    plt.title("length_std: " + str(length_std) + " length_mean: " + str(length_mean))
    plt.savefig(png_path)


class data(object):
    def __init__(self, file_test):
        self.df = pd.read_csv(file_test).dropna()
        self.context = self.df["comment_text"].values
        self.label = self.df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
        self.label = [list(idx) for idx in self.label]

        print(type(self.label[0]))
        print(np.array(self.label).shape)
        # self.normal_label = np.array([0 if number > 0 else 1 for number in np.sum(self.label, axis=1)])
        # self.label = np.concatenate((self.label, np.expand_dims(self.normal_label, axis=-1)), axis=1)

    def encode_word(self):
        path = os.path.dirname(__file__)
        if not os.path.exists(os.path.join(path, "data/pkl/train.pkl")):
            vocab_processor = learn.preprocessing.VocabularyProcessor(
                max_document_length=300,
                min_frequency=5
            )
            self.context = [self.text_to_wordlist(article) for article in self.context]
            all_context = list(vocab_processor.fit_transform(self.context))
            context_ids = [list(range(len(vocab_processor.vocabulary_)))]

            print("number of words :", len(vocab_processor.vocabulary_))
            print(vocab_processor.reverse(context_ids))
            for index in vocab_processor.reverse(context_ids):
                print(index)

            self.train_x, self.test_x, self.train_y, self.test_y = \
                train_test_split(list(all_context), self.label, test_size=0.02)

            print(os.path.join(path, "data/pkl/train.pkl"))
            print(os.path.join(path, "data/pkl/test.pkl"))
            print(os.path.join(path, "data/pkl/bag.pkl"))

            print(np.sum(self.test_y, axis=0))

            pickle.dump((self.train_x, self.train_y), open(os.path.join(path, "data/pkl/train.pkl"), "wb"))
            pickle.dump((self.test_x, self.test_y), open(os.path.join(path, "data/pkl/test.pkl"), "wb"))
            pickle.dump(list(vocab_processor.reverse(context_ids)), open(os.path.join(path, "data/pkl/bag.pkl"), "wb"))
        else:
            self.train_x, self.train_y = pickle.load(open(os.path.join(path, "data/pkl/train.pkl"), "rb"))
            self.test_x, self.test_y = pickle.load(open(os.path.join(path, "data/pkl/test.pkl"), "rb"))
        return self

    @staticmethod
    def text_to_wordlist(text):
        text = re.sub(r"[^A-Za-z0-9]", " ", text)
        text = re.sub(r"what's", "", text)
        text = re.sub(r"What's", "", text)
        text = re.sub(r"\'s", " ", text)
        text = re.sub(r"\'ve", " have ", text)
        text = re.sub(r"can't", "cannot ", text)
        text = re.sub(r"n't", " not ", text)
        text = re.sub(r"I'm", "I am", text)
        text = re.sub(r" m ", " am ", text)
        text = re.sub(r"\'re", " are ", text)
        text = re.sub(r"\'d", " would ", text)
        text = re.sub(r"\'ll", " will ", text)
        text = re.sub(r"60k", " 60000 ", text)
        text = re.sub(r" e g ", " eg ", text)
        text = re.sub(r" b g ", " bg ", text)
        text = re.sub(r"\0s", "0", text)
        text = re.sub(r" 9 11 ", "911", text)
        text = re.sub(r"e-mail", "email", text)
        text = re.sub(r"\s{2,}", " ", text)
        text = re.sub(r"quikly", "quickly", text)
        text = re.sub(r" usa ", " America ", text)
        text = re.sub(r" USA ", " America ", text)
        text = re.sub(r" u s ", " America ", text)
        text = re.sub(r" uk ", " England ", text)
        text = re.sub(r" UK ", " England ", text)
        text = re.sub(r"india", "India", text)
        text = re.sub(r"switzerland", "Switzerland", text)
        text = re.sub(r"china", "China", text)
        text = re.sub(r"chinese", "Chinese", text)
        text = re.sub(r"imrovement", "improvement", text)
        text = re.sub(r"intially", "initially", text)
        text = re.sub(r"quora", "Quora", text)
        text = re.sub(r" dms ", "direct messages ", text)
        text = re.sub(r"demonitization", "demonetization", text)
        text = re.sub(r"actived", "active", text)
        text = re.sub(r"kms", " kilometers ", text)
        text = re.sub(r"KMs", " kilometers ", text)
        text = re.sub(r" cs ", " computer science ", text)
        text = re.sub(r" upvotes ", " up votes ", text)
        text = re.sub(r" iPhone ", " phone ", text)
        text = re.sub(r"\0rs ", " rs ", text)
        text = re.sub(r"calender", "calendar", text)
        text = re.sub(r"ios", "operating system", text)
        text = re.sub(r"gps", "GPS", text)
        text = re.sub(r"gst", "GST", text)
        text = re.sub(r"programing", "programming", text)
        text = re.sub(r"bestfriend", "best friend", text)
        text = re.sub(r"dna", "DNA", text)
        text = re.sub(r"III", "3", text)
        text = re.sub(r"the US", "America", text)
        text = re.sub(r"Astrology", "astrology", text)
        text = re.sub(r"Method", "method", text)
        text = re.sub(r"Find", "find", text)
        text = re.sub(r"banglore", "Banglore", text)
        text = re.sub(r" J K ", " JK ", text)
        return text

    @staticmethod
    def get_batch(epoches, batch_size):
        path = os.path.dirname(__file__)
        train_x, train_y = pickle.load(open(os.path.join(path, "data/pkl/train.pkl"), "rb"))
        data = list(zip(train_x, train_y))

        for epoch in range(epoches):
            random.shuffle(data)
            for batch in range(0, len(train_y), batch_size):
                if batch + batch_size < len(train_y):
                    yield data[batch: (batch + batch_size)]


count_information()

if __name__ == "__main__":
    path = os.path.dirname(__file__)
    # Data = data(os.path.join(path, "data/csv/train.csv")).encode_word()
