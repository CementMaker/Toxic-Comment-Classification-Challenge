import tensorflow as tf
import datetime
import pickle
import numpy as np
import sys

sys.path.append('../')

from cnn import *
from PreProcess import *
from sklearn.metrics import auc
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score

from tensorflow import metrics

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
        lstm = Cnn(sequence_length=300,
                   embedding_size=100,
                   vocab_size=45779,
                   num_filters=100,
                   filter_sizes=[1, 2, 3, 4, 5],
                   num_classes=6)

        optimizer = tf.train.AdamOptimizer(0.001)
        grads_and_vars = optimizer.compute_gradients(lstm.losses)
        global_step = tf.Variable(0, name="global_step", trainable=False)
        tr_op_set = optimizer.apply_gradients(grads_and_vars, global_step=global_step)
        sess.run(tf.global_variables_initializer())

        def train_step(batch, label):
            feed_dict = {
                lstm.input_sentence: batch,
                lstm.label: label,
            }

            time_str = datetime.datetime.now().isoformat()
            _, step, loss, score = sess.run([tr_op_set, global_step, lstm.losses, lstm.score], feed_dict=feed_dict)
            print("{}: step {}, loss {}".format(time_str, step, loss))

        def dev_step(batch, label):
            feed_dict = {
                lstm.input_sentence: batch,
                lstm.label: label,
            }
            step, loss, score = sess.run([global_step, lstm.losses, lstm.score], feed_dict=feed_dict)
            time_str = datetime.datetime.now().isoformat()

            print("{}: step {}, loss {}, auc {}".format(time_str, step, loss, roc_auc_score(y_true=np.array(label),
                                                                                            y_score=np.array(score))))


        Batch = data(os.path.join(os.path.dirname(__file__), "../data/csv/train.csv")).encode_word().get_batch(2, 200)
        x_dev, y_dev = pickle.load(open(os.path.join(os.path.dirname(__file__), "../data/pkl/test.pkl"), "rb"))
        for data in Batch:
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

