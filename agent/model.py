import tensorflow as tf

#TODO don't forget to set unavailable policy actions to 0 and rescale
class FullyConv:
    def __init__(self, num_actions, use_lstm, input_data_format, use_data_format='channels_first'):
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.input_data_format = input_data_format
        self.use_data_format = use_data_format

    # Build and return the policy and value TensorFlow operations
    def build(self, screen_input, minimap_input, nonspatial_input):
        # Convert to the correct data format if needed
        if self.input_data_format != self.use_data_format:
            screen_input = self._switch_data_format(screen_input)
            minimap_input = self._switch_data_format(minimap_input)

        #

    #
    def sample_action(self, policy):
        print('Sample action')

    # Convert to the other data format
    def _switch_data_format(self, input_array):
        if self.input_data_format == 'channels_last':
            # Convert from NHWC to NCHW
            return tf.transpose(input_array, [0, 3, 1, 2])
        else:
            # Convert from NCHW to NHWC
            return tf.transpose(input_array, [0, 2, 3, 1])
