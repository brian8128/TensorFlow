from model_params import Params
from model import RegressionModel
from data import get_batch

import tensorflow as tf
import numpy as np
import os
import matplotlib.pyplot as plt
import cPickle as pickle


def main():

    params = Params()

    model = RegressionModel(params)

    # Use functions of the model to build the graph

    out, states = model.inference(model.data_placeholder)
    loss = model.loss(out, model.labels_placeholder)
    train_op = model.train(loss, params.step_size)

    # Create a session for running Ops on the Graph.
    sess = tf.Session()

    # Run the Op to initialize the variables.
    init = tf.initialize_all_variables()
    sess.run(init)
    saver = tf.train.Saver(tf.all_variables())

    for i in range(params.train_steps + 1):
        data, labels = get_batch(params.batch_size, params.sequence_length, params.input_channels)
        feed_dict = {
            model.data_placeholder: data,
            model.labels_placeholder: labels
        }

        # Run one step of the model.  The return values are the activations
        # from the `train_op` (which is discarded) and the `loss` Op.  To
        # inspect the values of your Ops or variables, you may include them
        # in the list passed to sess.run() and the value tensors will be
        # returned in the tuple from the call.
        _, loss_value = sess.run([train_op, loss], feed_dict=feed_dict)
        if i % params.print_every == 0:
            print i, loss_value
        if i % params.save_every == 0:
            name = "model_{0}.ckpt".format(params.get_id)
            checkpoint_path = os.path.join('./save', name)
            # TODO: If we restore a model for further training, we should
            # add the number of training steps it had completed to our global step here
            saver.save(sess, checkpoint_path, global_step=i)
            print "model saved to {0}-{1}".format(checkpoint_path, i)
            with open('./save/{0}.model_param'.format(params.get_id), 'w') as f:
                pickle.dump(params,
                            f,
                            protocol=2 # pickle.HIGHEST_PROTOCOL as of writing
                            )

    data, labels = get_batch(params.batch_size, params.sequence_length, params.input_channels)
    feed_dict = {
            model.data_placeholder: data,
            model.labels_placeholder: labels
    }

    vars = sess.run(out + states, feed_dict)
    out_ = vars[0:len(out)]
    states_ = vars[len(out)+1:]

    d = data[0,0,:]
    o = np.array(out_)[:, 0, 0]
    l = labels[0,0,:]

    x1 = range(d.shape[0])
    x2 = range(1, d.shape[0] + 1)
    # TODO: output graph every 100 steps
    plt.scatter(x1, d, c='r')
    plt.scatter(x2, o, c='g')
    plt.scatter(x2, l, c='b', alpha=0.5)
    plt.show()

    print "data third dim", d

    print "out", o
    # print "states", np.array(states_)

    print "labels third dim", l


if __name__ == '__main__':
    main()
