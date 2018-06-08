import json
import unittest
import tensorflow as tf
from agent.modifiers.agent_modifier import AgentModifier
from agent.agent import A2CAgent
from pysc2.lib import actions
from environment.environment import Environment

class AgentTest(unittest.TestCase):

    def setUp(self):
        self.json_data = '{"observations": {"screen_features": ["height_map", "player_id", "player_relative", "unit_type"], ' \
                    '"minimap_features": ["player_id", "selected"], "nonspatial_features": ["player", "score_cumulative"], ' \
                    '"action_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}, "rewards": [1, 1, 1, 1]}'
        self.config = json.loads(self.json_data)
        self.sess = tf.Session()
        self.agent_modifier = AgentModifier(self.config, 32)
        self.agent = A2CAgent(self.sess, self.agent_modifier)
        # self.obs_spec = {}
        # self._builder = dummy_observation.Builder(self._obs_spec)
        # self.obs = self._builder.build().observation
        self.env = Environment()
        self.obs = self.env.reset()

    def testMakeAction(self):
        print("Testing Make Action")
        action = self.agent.act(self.obs)
        action_made_1 = self.agent.convert_actions(action)
        action_2 = self.agent.act(self.obs)
        self.obs = self.env.reset()
        action_made_2 = self.agent.convert_actions(action_2)
        self.assertNotEqual(action_made_1, action_made_2)

    def testGetObservationFeed(self):
        print("Testing Get Observation Feed")
        feed_dict = self.agent._get_observation_feed(self.obs)
        self.obs = self.env.reset()
        feed_dict_2 = self.agent._get_observation_feed(self.obs)
        self.assertNotEqual(feed_dict, feed_dict_2)

