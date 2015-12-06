from numpy import base_repr

# TODO: put params in a frozen dict
# Use the hash as the model id
# implement the __getattr__ function so we can get attributes and only set them once
# take a dict as an argument instead of all the individual params


class Params(object):

    def __init__(self,
                    input_channels=1,
                    output_channels=1,
                    rnn_size=16, # Number of hidden nodes
                    batch_size=30,  # Number of distinct sequences in a batch
                    sequence_length=30,  # Length of each sequence in the batch
                    step_size=0.001,
                    train_steps=2000,
                    print_every=100,
                    grad_clip=5,
                    parent_model_id=None, # If not none we load another model and continue training it
                    save_every=500
                 ):
        self.input_channels = input_channels
        self.output_channels = output_channels
        self.rnn_size = rnn_size # Number of hidden nodes
        self.batch_size = batch_size  # Number of distinct sequences in a batch
        self.sequence_length = sequence_length  # Length of each sequence in the batch
        self.step_size = step_size
        self.train_steps = train_steps
        self.print_every = print_every
        self.grad_clip = grad_clip
        self.save_every = save_every
        self.parent_model_id = parent_model_id

    @property
    def get_id(self):
        """
        A unique id to identify the specific instance of the model
        Guaranteed to probably be different for two different sets
        of params
        """
        return base_repr(hash(tuple(sorted(self.__dict__.items()))), 36).lower()[-6:]
