# -*- coding:utf-8 -*-
# !/usr/bin/env python

import pickle
import os
import jieba
import jieba.analyse
import random

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.model_selection import train_test_split
from tensorflow.contrib import learn
from collections import Counter
from gensim.models.word2vec import Word2Vec


def count_information():
    df = pd.read_csv("../data/train.csv").dropna()
    context = df["comment_text"].values
    label = df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values

    length = []
    for index in context:
        length.append(len(index.split()))

    Dict = sorted(Counter(length).items(), key=lambda val: val[0], reverse=False)
    key, value = zip(*Dict)
    plt.plot(key, value)
    plt.savefig("length.png")


class data(object):
    def __init__(self, file_test):
        self.df = pd.read_csv(file_test).dropna()
        self.context = self.df["comment_text"].values
        self.label = self.df[["toxic", "severe_toxic", "obscene", "threat", "insult", "identity_hate"]].values
        self.normal_label = np.array([0 if number > 0 else 1 for number in np.sum(self.label, axis=1)])
        self.label = np.concatenate((self.label, np.expand_dims(self.normal_label, axis=-1)), axis=1)

    def encode_word(self):
        if not os.path.exists("./pkl/train.pkl"):
            vocab_processor = learn.preprocessing.VocabularyProcessor(
                max_document_length=600,
                min_frequency=3
            )
            all_context = list(vocab_processor.fit_transform(self.context))
            context_ids = [list(range(len(vocab_processor.vocabulary_)))]

            print("number of words :", len(vocab_processor.vocabulary_))
            print(vocab_processor.reverse(context_ids))
            for index in vocab_processor.reverse(context_ids):
                print(index)

            self.train_x, self.test_x, self.train_y, self.test_y = train_test_split(list(all_context), self.label, test_size=0.01)
            pickle.dump((self.train_x, self.train_y), open("./pkl/train.pkl", "wb"))
            pickle.dump((self.test_x, self.test_y), open("./pkl/test.pkl", "wb"))
            pickle.dump(list(vocab_processor.reverse(context_ids)), open("./pkl/bag.pkl", "wb"))
        else:
            self.train_x, self.train_y = pickle.load(open("./pkl/train.pkl", "rb"))
            self.test_x, self.test_y = pickle.load(open("./pkl/test.pkl", "rb"))
        return self

    def load_word2vec(self):
        '''
        
        '''
        model = Word2Vec.load("../word2vec/model/word2vec.model")
        print("load models end")
        model.train(sentences=data.generator(self.context),
                    total_examples=model.corpus_count,
                    epochs=model.iter)
        print(model.corpus_count)
        print("train models end")
        print(model.most_similarity("model"))

    @staticmethod
    def generator(List):
        for index in List:
            yield index

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
    count_information()
    Data = data("../data/train.csv").encode_word().load_word2vec()
