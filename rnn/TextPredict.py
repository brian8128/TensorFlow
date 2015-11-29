from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import time
import tensorflow.python.platform
import numpy as np
import tensorflow as tf
from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import rnn
import random

"""
Idea is to train a basic rnn with LSTM to predict characters in Dr. Seuss.
"""

class TextPredict(object):

    def __init__(self, is_training):

        # Need to define self._train_op
        self.batch_size = batch_size = 50
        self.num_steps = num_steps = 1000
        self.hidden_size = 5000
        self.keep_prob = 0.5
        #self.num_layers = 2
        self._input_data = tf.placeholder(tf.int8, [256, batch_size, num_steps])
        self._targets = tf.placeholder(tf.int8, [256, batch_size, num_steps])

        logging = tf.logging

        lstm_cell = rnn_cell.BasicLSTMCell(self.hidden_size, forget_bias=0.0)
        if is_training and self.keep_prob < 1:
            lstm_cell = rnn_cell.DropoutWrapper(
                lstm_cell, output_keep_prob=self.keep_prob)
            #self._cell = cell = rnn_cell.MultiRNNCell([lstm_cell] * self.num_layers)
            self._cell = cell = lstm_cell


        # We need to define self._train_op.
        # Read num_steps chunks of data in.  Each chunk is length batch_size
        # and each data point is a one-hot vector of length 256
        # Compute the outputs using the RNN
        # Define the loss function as cross entropy
        # Tell an optimizer to reduce the loss


        #outputs, states = rnn.rnn(cell, inputs, initial_state=self._initial_state)



    def generate_text(self, n, session):
        """
        Generates and returns n characters of text in the style of the training data.
        """

        ## Somehow we have to ask the TF session to run some predictions for us.

        z = tf.constant(np.zeros(256))
        out = session.run(tf.nn.softmax(self._cell(z)))
        print("output:")
        for i in out:
            print(i)

    @property
    def train_op(self):
        return self._train_op




def get_batch(data, data_len, batch_size):
        """
        Randomly read a chunk of data out of self.data fo size batch size
        :return:
        """
        start = random.randint(data_len - batch_size)
        batch = data[start, start + batch_size]
        return one_hot(batch, 256)


def one_hot(label_batch, num_labels=256):
    """
    Translates dense labels into one hot encoding.  For example if we're dealing with
    numbers 1-4 we'd have

    1 -> 1, 0, 0, 0
    2 -> 0, 1, 0, 0
    3 -> 0, 0, 1, 0
    4 -> 0, 0, 0, 1

    :param label_batch: label_batch is a list of ints to process
    :param num_labels: 0 <= label < num_labels
    :return:
    """
    one_hot_labels = np.zeros((len(label_batch), 256))
    for i, l in enumerate(label_batch):
        one_hot_labels[i, l] = 1

    return one_hot_labels


def int_from_one_hot(label_batch, num_labels=256):
    int_labels = [0] * label_batch.shape[0]
    for i in range(label_batch.shape[0]):
        int_labels[i] = np.nonzero(label_batch[i, :])[0][0]
    return int_labels


def main(unused_args):

  with open("../data/seuss.txt") as f:
    data = f.read()
    data_len = len(data)

  with tf.Graph().as_default(), tf.Session() as session:
    initializer = tf.random_uniform_initializer(-0.1, 0.1)


if __name__ == "__main__":
  tf.app.run()