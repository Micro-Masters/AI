import tensorflow as tf
import numpy as np


class A2CAgent:
    def __init__(
            self, sess, model, discount=.99, value_loss_coeff=.5,
            entropy_loss_coeff=.001, learning_rate=1e-4, max_gradient_norm=1):
        print('A2C init')
        self.sess = sess
        self.model = model
        self.discount = discount #TODO: use discount from sc2_env observation instead, won't need this class var then
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm
        self.value_loss_coeff = value_loss_coeff
        self.entropy_loss_coeff = entropy_loss_coeff
        self._build()

    # Build and initialize the TensorFlow graph
    def _build(self):
        self.action, self.value = self._build_model()
        loss = self._build_loss()

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.99, epsilon=1e-5)

        self.train_operation = tf.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            learning_rate=None,
            optimizer=optimizer,
            clip_gradients=self.max_gradient_norm,
            name="train_operation")

    # Build the TensorFlow action (single action from policy) & value Tensors
    def _build_model(self):
        policy, value = self.model.build()
        #TODO sample action from policy
        return action, value

    # Build the TensorFlow loss tensor
    def _build_loss(self):
        # Tensorflow placeholders
        returns = tf.placeholder(tf.float32, [None], 'returns')
        values = tf.placeholder(tf.float32, [None], 'values')
        advantages = tf.stop_gradient(returns - values)

        # Calculate loss Tensor using placeholders
        negative_log_policy = #TODO
        policy_loss = tf.reduce_mean(advantages * negative_log_policy)
        value_loss = self.value_loss_coeff * tf.losses.reduce_mean_squared_error(values, returns)
        entropy_loss = self.entropy_loss_coeff * #TODO

        loss = policy_loss + value_loss - entropy_loss
        return loss

    def act(self, observation):
        # TODO: Pass observation into feed_dict below
        return self.sess.run([self.action, self.value])

    # Train the model to minimize loss
    def train(self, observations, actions, rewards, dones, values, next_value):
        returns = self.get_returns(rewards, dones, values, next_value)
        self.sess.run(self.train_operation)

    # Compute return values as return = reward + discount * next value * (0 if done else 1)
    def get_returns(self, rewards, dones, values, next_value):
        returns = np.zeros((rewards.shape[0], rewards.shape[1]), dtype=np.float32)
        # Compute the last return using next_value
        returns[-1] = rewards[-1] + self.discount * next_value * (1 - dones[-1])
        # Compute the rest in a loop using values, working backward
        for i in reversed(range(returns.size[0] - 1)):
            returns[i] = rewards[i] + self.discount * values[i+1] * (1 - dones[i])
        return returns
