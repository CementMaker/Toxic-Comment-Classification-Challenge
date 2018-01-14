# -*- coding:utf-8 -*-
# !/usr/bin/env python

import pickle
import os
import jieba
import jieba.analyse
import random

import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from collections import Counter


def count_information():
    df = pd.read_csv("../data/train.csv").dropna()
    context = df["comment_text"].values
    label = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    length = []
    for index in context:
        length.append(len(index.split()))

    Dict = sorted(Counter(length).items(), key=lambda val: val[0], reverse=False)
    key, value = zip(*Dict)
    print(Dict)
    plt.plot(key, value)
    plt.savefig("length.png")


class data(object):
    def __init__(self, file_test):
        self.df = pd.read_csv(file_test).dropna()
        self.context = self.df["comment_text"].values
        self.label = self.df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    def encode_word(self):
        if os.path.exists("./pkl/train.csv"):
            vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length=600, min_frequency=5)
            all_context = list(vocab_processor.fit_transform(self.context))
            print("number of words :", len(vocab_processor.vocabulary_))
            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(list(all_context), self.label, test_size=0.01)
            pickle.dump((self.train_x, self.train_y), open("./pkl/train.pkl", "wb"))
            pickle.dump((self.test_x, self.test_y), open("./pkl/test.pkl", "wb"))
        else:
            self.train_x, self.train_y = pickle.load(open("./pkl/train.pkl", "rb"))
            self.test_x, self.test_y = pickle.load(open("./pkl/test.pkl", "rb"))
        return self

    @staticmethod
    def get_batch(epoches, batch_size):
        train_x, train_y = pickle.load(open("./pkl/train.pkl", "rb"))
        data = list(zip(train_x, train_y))

        for epoch in range(epoches):
            random.shuffle(data)
            for batch in range(0, len(train_y), batch_size):
                if batch + batch_size < len(train_y):
                    yield data[batch: (batch + batch_size)]


if __name__ == "__main__":
    Data = data("../data/train.csv").encode_word().get_batch(5, 400)

    number = 0
    for idx in Data:
        number += 1
    print(number)

