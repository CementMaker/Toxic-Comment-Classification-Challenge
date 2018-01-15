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
                    vocab_size=47000,
                    num_label=7,
                    batch_size=800)

        optimizer = tf.train.AdamOptimizer(0.001)
        grads_and_vars = optimizer.compute_gradients(lstm.loss_train)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)

        sess.run(tf.global_variables_initializer())

        def train_step(batch, label):
            feed_dict = {
                lstm.input: batch,
                lstm.label: label,
            }
            _, step, loss = sess.run([tr_op_set, global_step, lstm.loss_train], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {}".format(time_str, step, loss))
            global_loss.append(loss)


        def dev_step(batch, label):
            feed_dict = {
                lstm.input: batch,
                lstm.label: label,
            }
            step, loss = sess.run([global_step, lstm.loss_test], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()
            print("{}: step {}, loss {:g}".format(time_str, step, loss))

        batches = data("../data/train.csv").encode_word().get_batch(10, 800)
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

