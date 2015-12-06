import tensorflow as tf

import argparse
import cPickle as pickle
import glob
import numpy as np

from model import RegressionModel

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--save_dir', type=str, default='save',
                       help='model directory to store checkpointed models')
    parser.add_argument('-n', type=int, default=3,
                       help='number of characters to sample')
    parser.add_argument('--prime', type=list, default=[1.5, 2, 2.5, 3, 3.5, 4, 4.5, 5, 5.5, 6],
                       help='prime arr')
    parser.add_argument('--params', type=str, default='tycf0h',
                        help='the six character model_params id, ex: wvs1bp')
    args = parser.parse_args()
    sample(args)

def sample(args):
    # Need to load the model from somewhere
    params = pickle.load(open('./save/{0}.model_param'.format(args.params)))

    model = RegressionModel(params, infer=True)
    model.inference(model.data_placeholder)

    with tf.Session() as sess:
        # TODO: This loads the most recent checkpoint not
        tf.initialize_all_variables().run()
        saver = tf.train.Saver(tf.all_variables())
        checkpoints = glob.glob("./save/model_{0}.ckpt-*".format(params.id))
        if len(checkpoints) > 0:
            # We have worked on training this model before. Resume work rather than
            # starting from scratch

            # Get the iteration number for all of them
            iterations = np.array([int(c.split('-')[1]) for c in checkpoints])
            # Index of the checkpoint with the most iterations
            idx = np.argmax(iterations)

            restore_path = checkpoints[idx]
            saver.restore(sess, restore_path)
            print "restoring {0}".format(restore_path)
            print model.sample(sess, args.n, args.prime)
        else:
            print "Unable to restore - no checkpoints found"


if __name__ == '__main__':
    main()
