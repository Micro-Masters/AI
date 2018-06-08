import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from pysc2.lib import actions, features


#TODO don't forget to set unavailable policy actions to 0 and rescale
class FullyConv:
    def __init__(self, num_actions, use_lstm, input_data_format, use_data_format='NCHW'):
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.input_data_format = input_data_format
        self.use_data_format = use_data_format

    # Build and return the policy and value TensorFlow operations
    def build(self, screen_input, minimap_input, nonspatial_input, available_actions,
              screen_feature_names, minimap_feature_names, nonspatial_feature_names):
        # Convert to the correct data format if needed
        if self.input_data_format != self.use_data_format:
            screen_input = self._switch_data_format(screen_input)
            minimap_input = self._switch_data_format(minimap_input)

        # Build state
        size = screen_input.shape[2]
        screen = self._build_cnn_block(screen_input, features.SCREEN_FEATURES, screen_feature_names)
        minimap = self._build_cnn_block(minimap_input, features.MINIMAP_FEATURES, minimap_feature_names)
        nonspatial = self._build_nonspatial_block(nonspatial_input, size, nonspatial_feature_names)
        state = tf.concat([screen, minimap, nonspatial], axis=1)

        # Get outputs
        nonspatial_out = layers.fully_connected(layers.flatten(state), num_outputs=256)
        value = np.reshape(layers.fully_connected(nonspatial_out, 1, activation_fn=None), [-1])
        #TODO spatial and action policy

        return policy, value

    # Process/build spatial feature layers
    def _build_cnn_block(self, input, all_features, our_features):
        feature_layers = tf.split(input, input.shape[1], axis=1)
        for i, feat in enumerate(feature_layers):
            feature_type = getattr(all_features, our_features[i]).type
            scale = getattr(all_features, our_features[i]).scale
            if feature_type == features.FeatureType.CATEGORICAL:
                feature_layers[i] = tf.one_hot(tf.to_int32(tf.squeeze(feature_layers[i], axis=1)), scale, axis=1)
                feature_layers[i] = layers.conv2d(feature_layers[i], num_outputs=np.log2(scale),
                                                  kernel_size=1, data_format=self.use_data_format)
            else:
                feature_layers[i] = tf.log(feature_layers[i] + 1.0)

        block = tf.concat(features, axis=1)
        convolution1 = layers.conv2d(block, num_outputs=16, kernel_size=5, data_format="NCHW")
        convolution2 = layers.conv2d(convolution1, num_outputs=32, kernel_size=3, data_format="NCHW")
        return convolution2

    # Process/build nonspatial features
    def _build_nonspatial_block(self, input, size, names):
        # TODO: add more in later, 0:player
        return self.broadcast_features(tf.log(input[0] + 1.0), size)

    # Broadcast tensor along channels
    def broadcast_features(self, features, sz):
        return tf.tile(tf.expand_dims(tf.expand_dims(features, 2), 3), [1, 1, sz, sz])

    # Select a single action from the policy
    def sample_action(self, policy):
        #TODO
        print('Sample action')

    # Convert to the other data format
    def _switch_data_format(self, input_array):
        if self.input_data_format == 'NHWC':
            # Convert from NHWC to NCHW
            return tf.transpose(input_array, [0, 3, 1, 2])
        else:
            # Convert from NCHW to NHWC
            return tf.transpose(input_array, [0, 2, 3, 1])
