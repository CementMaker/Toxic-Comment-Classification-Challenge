import tensorflow as tf


class nbow(object):
    def __init__(self, sentence_len, vocab_size, embedding_size, num_label, batch_size):
        self.input = tf.placeholder(dtype=tf.int32, shape=[None, sentence_len])
        self.label = tf.placeholder(dtype=tf.float32, shape=[None, num_label])

        with tf.name_scope("embedding"):
            w = tf.Variable(tf.truncated_normal(shape=[vocab_size, embedding_size]), dtype=tf.float32)
            self.input_embedding = tf.nn.embedding_lookup(params=w, ids=self.input)

        with tf.name_scope("mean_pooling"):
            self.feature = tf.reduce_mean(self.input_embedding, axis=1)

        with tf.name_scope("full_connected"):
            w = tf.Variable(tf.truncated_normal(shape=[embedding_size, num_label]), dtype=tf.float32)
            b = tf.Variable(tf.truncated_normal(shape=[num_label]), dtype=tf.float32)
            self.logits = tf.sigmoid(tf.nn.xw_plus_b(self.feature, w, b))

        with tf.name_scope("loss_train"):
            self.losses_train = tf.nn.softmax_cross_entropy_with_logits(logits=self.logits, labels=self.label)
            self.loss_train = tf.reduce_mean(self.losses_train)

        with tf.name_scope("loss_test"):
            self.logits_test = tf.slice(self.logits, [0, 0], [batch_size, num_label - 1])
            self.label_test = tf.slice(self.label, [0, 0], [batch_size, num_label - 1])
            self.losses_test = tf.nn.sigmoid_cross_entropy_with_logits(logits=self.logits_test, labels=self.label_test)
            self.loss_test = tf.reduce_mean(self.losses_test)


# lstm = nbow(sentence_len=600,
#             embedding_size=150,
#             vocab_size=47000,
#             num_label=7,
#             batch_size=100)
