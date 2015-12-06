import numpy as np


def get_batch(batch_size, sequence_length, input_channels, test=False):
    return get_sequence(batch_size, sequence_length, input_channels)

def get_sequence(batch_sz, length, input_channels):
    labels = np.zeros([batch_sz, input_channels, length + 1])
    for i in range(batch_sz):
        labels[i,0] = np.linspace(10 * np.random.random(), 10 * np.random.random(), num=length + 1)
    data = labels  + np.random.normal(size=[batch_sz, input_channels, length + 1]) / 100.
    # We want the RNN to predict the 'clean' labels based on the dirty data.
    # We also want it to predict one timestep in advance
    return data[:, :, :-1], labels[:, :, 1:]