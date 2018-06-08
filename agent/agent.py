import tensorflow as tf
import numpy as np
import pandas as pd

from agent.model import FullyConv


class A2CAgent:
    def __init__(
            self, sess, agent_modifier, use_lstm=False, discount=.99, value_loss_coeff=.5,
            entropy_loss_coeff=.001, learning_rate=1e-4, max_gradient_norm=1):
        self.sess = sess
        self.use_lstm = use_lstm
        self.agent_modifier = agent_modifier
        self.discount = discount #TODO: use discount from sc2_env observation instead, won't need this class var then
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self._build()

    # Build and initialize the TensorFlow graph
    def _build(self):
        self.action, self.value = self._build_model()
        self.train_operation = self._build_optimizer()

    # Build TensorFlow action (single action from policy) & value operations
    def _build_model(self):
        # Declare model inputs (of any number of observations) and initialize model
        observation_shapes = {v.insert(0, None)
                              for k, v in self.self.agent_modifier.observation_shapes.items()}
        self.screen_input = tf.placeholder(tf.float32, observation_shapes['screen'], 'screen input')
        self.minimap_input = tf.placeholder(tf.float32, observation_shapes['minimap'], 'minimap input')
        self.nonspatial_input = tf.placeholder(tf.float32, observation_shapes['nonspatial'], 'nonspatial input')
        self.available_actions_input = tf.placeholder(tf.float32, observation_shapes['available_actions'], 'available actions input')

        model = FullyConv(self.agent_modifier.num_actions, self.use_lstm, self.agent_modifier.observation_data_format)

        # Create final action and value operations using model
        policy, value = self.model.build(self.screen_input, self.minimap_input, self.nonspatial_input)
        policy_action = model.sample_action(policy)
        action = self.agent_modifier.make_action(policy_action)
        return action, value

    # Build the TensorFlow loss operation
    def _build_optimizer(self):
        # TensorFlow placeholders and compute advantages
        #TODO actions placeholder
        self.returns = tf.placeholder(tf.float32, [None], 'returns')
        advantages = tf.stop_gradient(self.returns - self.value)

        # Create loss TensorFlow operation using placeholders
        negative_log_policy = #TODO
        policy_loss = tf.reduce_mean(advantages * negative_log_policy)
        value_loss = self.value_loss_coeff * tf.losses.reduce_mean_squared_error(self.value, self.returns)
        entropy_loss = self.entropy_loss_coeff * #TODO

        # Create the final optimizer using loss
        loss = policy_loss + value_loss - entropy_loss
        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.99, epsilon=1e-5)
        return tf.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=None,
            optimizer=optimizer,
            clip_gradients=self.max_gradient_norm,
            name="train_operation")

    # Take an observation (n_envs length array) and return the action, value
    def act(self, observation):
        #TODO also return log prob for action?
        feed_dict = self._get_observation_feed(observation)
        return self.sess.run([self.action, self.value], feed_dict=feed_dict)

    # Train the model to minimize loss
    def train(self, observations, actions, rewards, dones, values, next_value):
        # TODO pass actions
        observations_feed = self._get_observation_feed(observations, train=True)
        feed_dict = {
            self.returns: self._get_returns(rewards, dones, values, next_value).flatten(),
            **observations_feed
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
            new_observation = self.agent_modifier.modify_observations(observation)
        # Convert array of dictionaries to array of arrays
        observation_input = pd.DataFrame(new_observation).values()
        return {
            self.screen_input: observation_input[0],
            self.minimap_input: observation_input[1],
            self.nonspatial_input: observation_input[2],
            self.available_actions: observation_input[3]
        }
