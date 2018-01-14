import tensorflow as tf
import datetime
import pickle
import numpy as np

from nbow import *
from PreProcess import *

import matplotlib.pyplot as plt

global_loss = []
global_accuracy = []


with tf.Graph().as_default():
    session_conf = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False
    )
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        lstm = nbow(sentence_len=600,
                    embedding_size=150,
                    vocab_size=35000,
                    num_label=6)

        optimizer = tf.train.AdamOptimizer(0.0001)
        grads_and_vars = optimizer.compute_gradients(lstm.loss)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        def train_step(batch, label):
            feed_dict = {
                lstm.input: batch,
                lstm.label: label,
                lstm.dropout_keep_prob: 0.8
            }
            _, step, loss = sess.run(
                [tr_op_set, global_step, lstm.loss],
                feed_dict=feed_dict)

            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}".format(time_str, step, loss))
            global_loss.append(loss)

        def dev_step(batch, label):
            feed_dict = {
                lstm.input: batch,
                lstm.label: label,
                lstm.dropout_keep_prob: 0.0
            }
            step, loss = sess.run([global_step, lstm.loss], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))


        # x_train, y_train, x_dev, y_dev = split_data()
        # batches = batch_iter(list(zip(x_train, y_train)), batch_size=200, num_epochs=50)

        batches = data("../data/train.csv").encode_word().get_batch(5, 400)
        x_dev, y_dev = pickle.load(open("./pkl/test.pkl", "rb"))
        for data in batches:
            x_train, y_train = zip(*data)
            train_step(x_train, y_train)
            current_step = tf.train.global_step(sess, global_step)
            if current_step % 30 == 0:
                print("\nEvaluation:")
                dev_step(x_dev, y_dev)
                print("")

        x = list(range(len(global_loss)))
        plt.plot(x, global_loss, 'r', label="loss")
        plt.xlabel("batches")
        plt.ylabel("loss")
        plt.savefig("loss_modify.png")
        plt.close()

        plt.plot(x, global_accuracy, 'b', label="accuracy")
        plt.xlabel("batches")
        plt.ylabel("accuracy")
        plt.savefig("accuracy.png")
        plt.close()

