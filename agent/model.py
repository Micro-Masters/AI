import numpy as np
import tensorflow as tf
from tensorflow.contrib import layers
from pysc2.lib import actions, features


class FullyConv:
    def __init__(self, num_actions, use_lstm, input_data_format, use_data_format='NCHW'):
        self.num_actions = num_actions
        self.use_lstm = use_lstm
        self.input_data_format = input_data_format
        self.use_data_format = use_data_format

    # Select a single action from the policy
    def sample_action(self, policy):
        fn_out, args_out = policy

        fn_u = tf.random_uniform(tf.shape(fn_out))
        fn_sample = tf.argmax(tf.log(fn_u) / fn_out, axis=1)

        args_sample = dict()
        for arg_type, arg_out in args_out.items():
            arg_u = tf.random_uniform(tf.shape(arg_out))
            arg_sample = tf.argmax(tf.log(arg_u) / arg_out, axis=1)
            args_sample[arg_type] = arg_sample

        policy_action = (fn_sample, args_sample)
        return policy_action


    # Build and return the policy and value TensorFlow operations
    def build(self, screen_input, minimap_input, nonspatial_input, available_mask_input,
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
        value_out = tf.squeeze(layers.fully_connected(nonspatial_out, num_outputs=1, activation_fn=None), axis=1)

        fn_out = layers.fully_connected(nonspatial_out, num_outputs=self.num_actions, activation_fn=tf.nn.softmax)
        fn_out = self._mask(fn_out, available_mask_input)

        args_out = dict()
        for arg_type in actions.TYPES:
            if self._is_spatial_arg(arg_type):
                logits = layers.conv2d(state, num_outputs=1, kernel_size=1, activation_fn=None, data_format="NCHW")
                arg_out = tf.nn.softmax(layers.flatten(logits))
            else:
                arg_out = layers.fully_connected(nonspatial_out, num_outputs=arg_type.sizes[0], activation_fn=tf.nn.softmax)
            args_out[arg_type] = arg_out

        policy = (fn_out, args_out)
        return policy, value_out

    # Process/build spatial feature layers
    def _build_cnn_block(self, input, all_features, our_features):
        feature_layers = tf.split(input, input.shape[1], axis=1)
        for i, feat in enumerate(feature_layers):
            feature_type = getattr(all_features, our_features[i]).type
            scale = getattr(all_features, our_features[i]).scale
            if feature_type == features.FeatureType.CATEGORICAL:
                feature_layers[i] = tf.one_hot(tf.to_int32(tf.squeeze(feature_layers[i], axis=1)), scale, axis=1)
                feature_layers[i] = layers.conv2d(feature_layers[i], num_outputs=max(1, round(np.log2(scale))),
                                                  kernel_size=1, data_format=self.use_data_format)
            else:
                feature_layers[i] = tf.log(feature_layers[i] + 1.0)

        block = tf.concat(feature_layers, axis=1)
        convolution1 = layers.conv2d(block, num_outputs=16, kernel_size=5, data_format="NCHW")
        convolution2 = layers.conv2d(convolution1, num_outputs=32, kernel_size=3, data_format="NCHW")
        return convolution2

    # Process/build nonspatial features
    def _build_nonspatial_block(self, input, size, names):
        # investigate adding more in later, 0:player
        return self._broadcast_features(tf.log(input[0] + 1.0), size)

    # Broadcast tensor along channels
    def _broadcast_features(self, nonspatial_features, size):
        return tf.tile(tf.expand_dims(tf.expand_dims(nonspatial_features, 2), 3), [1, 1, size, size])

    # Convert to the other data format
    def _switch_data_format(self, input_array):
        if self.input_data_format == 'NHWC':
            # Convert from NHWC to NCHW
            return tf.transpose(input_array, [0, 3, 1, 2])
        else:
            # Convert from NCHW to NHWC
            return tf.transpose(input_array, [0, 2, 3, 1])

    def _mask(self, policy, available_mask):
        policy *= available_mask
        return policy / tf.clip_by_value(tf.reduce_sum(policy, axis=1, keep_dims=True), 1e-12, 1.0)

    def _is_spatial_arg(self, name):
        return name == 'screen' or name == 'screen2' or name == 'minimap'

    def get_neg_log_prob(self, action, policy):
        action_fns, action_args = actions
        policy_fns, policy_args = policy

        fn_log_prob = self._compute_log_probs(policy_fns, action_fns)
        log_prob = fn_log_prob

        for arg_type in action_args.keys():
            action_arg = action_args[arg_type]
            policy_arg = policy_args[arg_type]
            arg_log_prob = self._compute_log_probs(policy_arg, action_arg)
            log_prob += arg_log_prob
        return log_prob * -1.0

    def _compute_log_prob(probs, labels):
        # Select arbitrary element for unused arguments (log probs will be masked)
        labels = tf.maximum(labels, 0)
        indices = tf.stack([tf.range(tf.shape(labels)[0]), labels], axis=1)
        return tf.log(tf.clip_by_value(tf.gather_nd(probs, indices), 1e-12, 1.0))

    def get_entropy(self, policy):
        fns, args = policy

        entropy = -tf.reduce_sum(tf.log(tf.clip_by_value(fns, 1e-12, 1.0)) * fns, axis=-1)
        for arg_type in args.keys():
            arg = args[arg_type]
            arg_entropy = -tf.reduce_sum(tf.log(tf.clip_by_value(arg, 1e-12, 1.0)) * arg, axis=-1)
            entropy += arg_entropy
        return entropy
