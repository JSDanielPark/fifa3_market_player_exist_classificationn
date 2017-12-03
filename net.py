import tensorflow as tf
import numpy as np
from common import input_size, \
    input_y, \
    input_x, \
    input_color_chanel, \
    model_save_path, \
    output_size

class Net:
    def __init__(self, session, learning_rate=0.001):
        self.session = session
        self.learning_rate = learning_rate
        self._build_network()
        self.saver = tf.train.Saver()

    def _build_network(self):
        self._keep_prob = tf.placeholder(tf.float32)

        self.X = tf.placeholder(tf.float32, [None, input_size])
        X_img = tf.reshape(self.X, [-1, input_x, input_y, input_color_chanel])
        self.Y = tf.placeholder(tf.float32, [None, output_size])

        # Layer1
        W1 = tf.Variable(tf.random_normal([3, 3, 3, 32], stddev=0.01))
        net = tf.nn.conv2d(X_img, W1, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(net)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
        net = tf.nn.dropout(net, keep_prob=self._keep_prob)

        # Layer2
        W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))
        net = tf.nn.conv2d(net, W2, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(net)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1],
                             strides=[1, 2, 2, 1], padding='SAME')
        net = tf.nn.dropout(net, keep_prob=self._keep_prob)

        # Layer3
        W3 = tf.Variable(tf.random_normal([3, 3, 64, 128], stddev=0.01))
        net = tf.nn.conv2d(net, W3, strides=[1, 1, 1, 1], padding='SAME')
        net = tf.nn.relu(net)
        net = tf.nn.max_pool(net, ksize=[1, 2, 2, 1], strides=[
            1, 2, 2, 1], padding='SAME')
        net = tf.nn.dropout(net, keep_prob=self._keep_prob)

        weight_shape = 9 * 18 * 128
        net = tf.reshape(net, [-1, weight_shape])

        # Layer4
        W4 = tf.get_variable("W4", shape=[weight_shape, 625],
                             initializer=tf.contrib.layers.xavier_initializer())
        b4 = tf.Variable(tf.random_normal([625]))
        net = tf.nn.relu(tf.matmul(net, W4) + b4)
        net = tf.nn.dropout(net, keep_prob=self._keep_prob)

        W5 = tf.get_variable("W5", shape=[625, 1],
                             initializer=tf.contrib.layers.xavier_initializer())
        b5 = tf.Variable(tf.random_normal([1]))

        # 예측
        hypothesis = tf.sigmoid(tf.matmul(net, W5) + b5)

        # 비용함수
        self.cost = -tf.reduce_mean(self.Y * tf.log(hypothesis) + (1 - self.Y) *
                               tf.log(1 - hypothesis))
        self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.cost)
        self.predicted = tf.cast(hypothesis > 0.5, dtype=tf.float32)
        self.accuracy = tf.reduce_mean(tf.cast(tf.equal(self.predicted, self.Y), dtype=tf.float32))

    def save(self):
        self.saver.save(self.session, model_save_path)

    def restore(self):
        self.saver.restore(self.session, model_save_path)

    def predict(self, state, keep_prop=1.0):
        x = np.reshape(state, [1, input_size])
        predict = self.session.run(self.predicted, feed_dict={self.X: x, self._keep_prob: keep_prop})
        return predict

    def calc_accuracy(self, x_stack, y_stack, keep_prop=1.0):
        return self.session.run([self.accuracy], feed_dict={
            self.X: x_stack,
            self.Y: y_stack,
            self._keep_prob: keep_prop
        })

    def update(self, x_stack, y_stack, keep_prop=0.7):
        return self.session.run([self.cost, self.optimizer, self.accuracy], feed_dict={
            self.X: x_stack,
            self.Y: y_stack,
            self._keep_prob: keep_prop
        })
