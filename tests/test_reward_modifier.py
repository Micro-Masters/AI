from pysc2.lib import actions
import numpy as np
import pytest
import json
import unittest
from agent.modifiers.reward_modifier import RewardModifier
from pysc2.tests import dummy_observation
from environment.environment import Environment

_NO_OP = actions.FUNCTIONS.no_op.id
_NO_OP_STEP = [actions.FunctionCall(_NO_OP, [])]
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

class RewardModifierTest(unittest.TestCase):

    def setUp(self):
        json_data = '{"observations": {"screen_features": ["height_map", "player_id", "player_relative", "unit_type"], ' \
                   '"minimap_features": ["player_id", "selected"], "nonspatial_features": ["player", "score_cumulative"], ' \
                   '"action_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}, "rewards": [1, 1, 1, 1]}'
        config = json.loads(json_data)
        self._reward_mod = RewardModifier(config["rewards"])
        self.old_obs = [None]
        self.zero_obs = [None]
        # self._obs_spec = {}
        # self._builder = dummy_observation.Builder(self._obs_spec)
        # self.obs = self._builder.build()
        self.env = Environment()
        # self.obs = observations = [None] * 16
        self.obs = self.env.reset()

    def testModifyZero(self):
        print("Testing Zero")
        reward_1 = self._reward_mod.modify(self.obs[0], 0, self.zero_obs[0])
        self.assertEqual(reward_1, 0)

    def testModifyReset(self):
        print("Testing Reset")
        reward_1 = self._reward_mod.modify(self.obs[0], 0, self.old_obs[0])
        self._old_obs = self.obs
        self.obs = self.env.reset()
        reward_2 = self._reward_mod.modify(self.obs[0], 0, self.old_obs[0])
        self.assertEqual(reward_1, reward_2)

    def testModifySelectArmy(self):
        print("Testing Select Army")
        reward_1 = self._reward_mod.modify(self.obs[0], 0, self.old_obs[0])
        self._old_obs = self.obs
        new_obs = self.env.step([actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])])
        reward_2 = self._reward_mod.modify(new_obs[0], 0, self.old_obs[0])
        self.assertEqual(reward_1, reward_2)

    def testModifyAttack(self):
        print("Testing Attack")
        reward_1 = self._reward_mod.modify(self.obs[0], 0, self.old_obs[0])
        self._old_obs = self.obs
        new_obs = self.env.step([actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])])
        new_obs = self.env.step([actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [20, 20]])])
        reward_2 = self._reward_mod.modify(new_obs[0], 0, self.old_obs[0])
        self.assertEqual(reward_1, reward_2)


