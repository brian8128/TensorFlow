import tensorflow as tf
import numpy as np

from tensorflow.models.rnn import rnn_cell
from tensorflow.models.rnn import seq2seq


class RegressionModel(object):

    def __init__(self, params, infer=False):
        self.params = params

        if infer:
            self.batch_size = batch_size = 1
            self.sequence_length = sequence_length = 1
        else:
            self.batch_size = batch_size = self.params.batch_size
            self.sequence_length = sequence_length = self.params.sequence_length

        cell1 = rnn_cell.LSTMCell(self.params.rnn_size,
                                  self.params.input_channels,
                                  use_peepholes=True)
        cell2 = rnn_cell.LSTMCell(self.params.rnn_size,
                                  cell1.output_size,
                                  use_peepholes=True,
                                  num_proj=params.output_channels)
        self.cell = cell = rnn_cell.MultiRNNCell([cell1, cell2])

        self.data_placeholder = tf.placeholder(tf.float32,
                                               shape=(batch_size, params.input_channels, sequence_length),
                                               name='data_placeholder')
        self.labels_placeholder = tf.placeholder(tf.float32,
                                                 shape=(batch_size, params.input_channels, sequence_length),
                                                 name='labels_placeholder')

        # Initial state of the LSTM memory.
        # To train or to leave as all zeros...that is the question.  get_variable means train, zeros means zeros.
        # To make this a trainable variable we'd want it to be *the same* initial state for every sequence
        # in a batch
        self.initial_state = cell.zero_state(batch_size, dtype=tf.float32)

    def inference(self, input_data):
        """
        Build out the graph enough to make predictions
        input_data - a batch of sequences to predict.  Tensor of size [batch_size, input_channels, sequence_length]
        :return: logits
        """

        inputs = tf.split(2, self.sequence_length, input_data)  # Slice up the input_data into a list
        inputs = [tf.squeeze(input_, squeeze_dims=[2]) for input_ in inputs]  # Get rid of the dim with size 1

        self.outputs, self.states = seq2seq.rnn_decoder(inputs, # decoder_inputs: a list of 2D Tensors [batch_size x cell.input_size]
                                              self.initial_state,
                                              self.cell,
                                              None,  # Loop fn
                                              scope='inference'  # Name scope
                                              )
        #TODO: cleanup organziation
        self.final_state = self.states[-1]
        self.final_output = self.outputs[-1]

        return self.outputs, self.states

    def sample(self, sess, num=5, prime=[1, 2, 3, 4]):
        state = self.cell.zero_state(1, tf.float32).eval()
        for point in prime[:-1]:
            x = np.zeros((1, 1, 1))
            x[0, 0, 0] = point
            feed = {self.data_placeholder: x, self.initial_state:state}
            [_, state] = sess.run([self.final_output, self.final_state], feed)

        ret = prime
        point = prime[-1]
        for n in xrange(num):
            x = np.zeros((1, 1, 1))
            x[0, 0, 0] = point
            feed = {self.data_placeholder: x, self.initial_state:state}
            [out, state] = sess.run([self.final_output, self.final_state], feed)
            ret.append(out[0][0])
            point = out[0][0]
        return ret

    def loss(self, outputs, target_data):
        """
        Build out the graph enough to calculate the loss.  Here we are doing regression so we care about l2 loss.
        output_data - a batch of sequences to predict.  Tensor of size [input_channels, batch_size, n]
        """

        # packed_target = tf.pack(targets)  # packs a list into a tensor
        packed_output = tf.transpose(tf.pack(outputs), perm=[1, 2, 0])

        difference = tf.sub(packed_output, target_data)  # Subtract the target from the actual output

        loss = tf.nn.l2_loss(difference)  # The l2 norm of difference

        return loss

    def train(self, loss, learning_rate):

        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars),
                self.params.grad_clip)
        # Create the optimizer with the given learning rate.
        # Tried Adam optimizer here, for some reason it's terrible
        optimizer = tf.train.GradientDescentOptimizer(learning_rate)

        train_op = optimizer.apply_gradients(zip(grads, tvars))

        return train_op

# IDEA: Train a rnn to output prediction and confidence interval when doing sequence regression.
# What loss function would we use to train this?



