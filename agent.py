from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

#import numpy

from pysc2.agents import base_agent
from pysc2.lib import actions
import tensorflow as tf


####################################################################################
"""A random agent for starcraft."""

"""remove later, this is for environment testing purposes only"""

class RandomAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""

  def step(self, obs):
    super(RandomAgent, self).step(obs)
    function_id = numpy.random.choice(obs.observation.available_actions)
    args = [[numpy.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
    return actions.FunctionCall(function_id, args)
##################################################################################


class A2CAgent:
    def __init__(self, sess, policy, discount=.99, learning_rate=1e-4, max_gradient_norm=1):
        print('A2C init')
        self.sess = sess
        self.policy = policy
        self.discount = discount
        self.learning_rate = learning_rate
        self.max_gradient_norm = max_gradient_norm

    def _build(self):
        self.action, self.value = self.policy.build()
        #loss =

        optimizer = tf.train.RMSPropOptimizer(learning_rate=self.learning_rate, decay=0.99, epsilon=1e-5)
        self.train_op = tf.layers.optimize_loss(
            loss=loss,
            global_step=tf.train.get_global_step(),
            optimizer=optimizer,
            clip_gradients=self.max_gradient_norm,
            learning_rate=None,
            name="train_op")

    def act(self, observation):
        #pass observation
        return self.sess.run([self.action, self.value])

    def train(self, observations, actions, rewards, values, dones):
        returns = get_returns()
        advantages = returns - values
        print('A2C train')

    def get_returns(self):
        print('get returns')