import tensorflow as tf
import numpy as np
from collections import defaultdict
from tensorflow.contrib import layers


from agent.model import FullyConv


class A2CAgent:
    def __init__(
            self, sess, agent_modifier, use_lstm=False, discount=.99, value_loss_coeff=.5,
            entropy_loss_coeff=.001, learning_rate=1e-4, max_gradient_norm=1.0):
        self.sess = sess
        self.use_lstm = use_lstm
        self.agent_modifier = agent_modifier
        self.discount = discount #Could use discount from sc2_env observation instead, won't need this class var then
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self._build()

    # Build and initialize the TensorFlow graph
    def _build(self):
        self.policy, self.action, self.value = self._build_model()
        self.train_operation = self._build_optimizer()
        self.sess.run(tf.global_variables_initializer())

    # Build TensorFlow action (single action from policy) & value operations
    def _build_model(self):
        # Declare model inputs (of any number of observations) and initialize model
        observation_shapes = self.agent_modifier.observation_shapes
        for k, v in observation_shapes.items():
            observation_shapes[k].insert(0, None)
        print(observation_shapes['screen'])
        self.screen_input = tf.placeholder(tf.float32, observation_shapes['screen'], 'screen_input')
        self.minimap_input = tf.placeholder(tf.float32, observation_shapes['minimap'], 'minimap_input')
        self.nonspatial_input = tf.placeholder(tf.float32, observation_shapes['nonspatial'], 'nonspatial_input')
        self.available_mask_input = tf.placeholder(tf.float32, observation_shapes['available_mask'], 'available_mask_input')

        self.model = FullyConv(self.agent_modifier.num_actions, self.use_lstm, self.agent_modifier.observation_data_format)

        # Create final action and value operations using model
        policy, value = self.model.build(self.screen_input, self.minimap_input, self.nonspatial_input,
                                    self.available_mask_input, *self.agent_modifier.feature_names)
        action = self.model.sample_action(policy)
        return policy, action, value

    # Build the TensorFlow loss operation
    def _build_optimizer(self):
        # TensorFlow placeholders and compute advantages
        fns = tf.placeholder(tf.int32, [None], 'fns')
        args = {
            k: tf.placeholder(tf.int32, [None], 'args')
            for k in self.policy[1].keys()}
        self.actions = (fns, args)
        self.returns = tf.placeholder(tf.float32, [None], 'returns')
        advantages = tf.stop_gradient(self.returns - self.value)

        # Create loss TensorFlow operation using placeholders
        negative_log_policy = self.model.get_neg_log_prob(self.actions, self.policy)
        policy_loss = tf.reduce_mean(advantages * negative_log_policy)
        value_loss = self.value_loss_coeff * tf.reduce_mean(tf.square(self.value - self.returns))
        entropy_loss = self.entropy_loss_coeff * tf.reduce_mean(self.model.get_entropy(self.policy))

        # Create the final optimizer using loss
        loss = policy_loss + value_loss - entropy_loss
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.99, epsilon=1e-5)
        return layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=None,
            optimizer=optimizer,
            clip_gradients=self.max_gradient_norm,
            name="train_operation")

    # Convert policy action syntax to a PySC2 FunctionCall action
    def convert_action(self, policy_action):
        return self.agent_modifier.make_action(policy_action)

    # Take an observation (n_envs length array) and return the action, value
    def act(self, observation):
        feed_dict = self._get_observation_feed(observation)
        action, value = self.sess.run([self.action, self.value], feed_dict=feed_dict)
        return action, value

    # Train the model to minimize loss
    def train(self, observations, actions, rewards, dones, values, next_value):
        observations_feed = self._get_observation_feed(observations, train=True)
        actions_feed = self._get_action_feed(actions)
        feed_dict = {
            self.returns: self._get_returns(rewards, dones, values, next_value).flatten(),
            **observations_feed,
            **actions_feed
        }
        self.sess.run(self.train_operation, feed_dict=feed_dict)

    # Compute return values as return = reward + discount * next value * (0 if done else 1)
    def _get_returns(self, rewards, dones, values, next_value):
        returns = np.zeros(rewards.shape, dtype=np.float32)
        # Compute the last return using next_value
        returns[-1] = rewards[-1] + self.discount * next_value * (1 - dones[-1])
        # Compute the rest in a loop using values, working backward
        for i in reversed(range(returns.size[0] - 1)):
            returns[i] = rewards[i] + self.discount * values[i+1] * (1 - dones[i])
        return returns

    # Modify PySC2 observations and then return the observation input feed
    def _get_observation_feed(self, observation, train=False):
        # Convert PySC2 observations to an array of dictionaries
        if train:
            new_observations = [self.agent_modifier.modify_observations(obs) for obs in observation]
            new_observation = new_observations.reshape(-1, *new_observations.shape[1:])
        else:
            new_observation = self.agent_modifier.modify_observation(observation)
        # Convert array of dictionaries to array of arrays
        observation_input = []
        for obs_dict in new_observation:
            obs = [np.array(v) for k, v in obs_dict.items()]
            observation_input.append(obs)
        observation_input = np.array(observation_input)
        print(observation_input.shape)
        observation_input = np.transpose(observation_input)
        print(observation_input.shape)
        print(len(observation_input[0]))

        return {
            self.screen_input: observation_input[0],
            self.minimap_input: observation_input[1],
            self.nonspatial_input: observation_input[2],
            self.available_mask_input: observation_input[3]
        }

    # Join actions into single fn list and single arg map
    def _get_action_feed(self, actions):
        fns = []
        args = defaultdict(list)

        for fn, arg in actions:
            fns.append(fn)
            for k, v in arg:
                args[k].append(v)
        return {self.actions: (np.array(fns).flatten(), args)}
