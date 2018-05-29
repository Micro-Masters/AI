import tensorflow as tf
import numpy as np
import math
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.lib import features
from absl import flags

_NO_OP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index
_PLAYER_ID = features.SCREEN_FEATURES.player_id.index

_PLAYER_SELF = 1
_PLAYER_HOSTILE = 0

_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

ACTION_DO_NOTHING = 'donothing'
ACTION_ATTACK = 'attack'

smart_actions = [
    ACTION_DO_NOTHING,
]

#edit for specific map
for mm_x in range(0, 64):
    for mm_y in range(0, 64):
        if (mm_x + 1) % 32 == 0 and (mm_y + 1) % 32 == 0:
            smart_actions.append(ACTION_ATTACK + '_' + str(mm_x - 16) + '_' + str(mm_y - 16))
###

print("here!!!")
print(len(smart_actions))
FLAGS = flags.FLAGS

class A2CAgent:
    def __init__(self, sess, policy, learning_rate=1e-4, max_gradient_norm=1):
        print('A2C init')
        self.sess = sess
        self.policy = policy
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

    def train(self):
        print('A2C train')

    def act(self, observation):
        #pass observation
        self.action = self.policy.getAction(observation)
        return self.sess.run([self.action, self.value])

    def transformObservation(self, obs):
        new_observation = np.zeros(8)
        hot_squares = np.zeros(4)        
        enemy_y, enemy_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_HOSTILE).nonzero()
        for i in range(0, len(enemy_y)):
            y = int(math.ceil((enemy_y[i] + 1) / 32))
            x = int(math.ceil((enemy_x[i] + 1) / 32))
                
            hot_squares[((y - 1) * 2) + (x - 1)] = 1
            
        for i in range(0, 4):
            new_observation[i] = hot_squares[i]
	
        green_squares = np.zeros(4)        
        friendly_y, friendly_x = (obs.observation['minimap'][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
        for i in range(0, len(friendly_y)):
            y = int(math.ceil((friendly_y[i] + 1) / 32))
            x = int(math.ceil((friendly_x[i] + 1) / 32))
                
            green_squares[((y - 1) * 2) + (x - 1)] = 1
            
        for i in range(0, 4):
            new_observation[i + 4] = green_squares[i]

