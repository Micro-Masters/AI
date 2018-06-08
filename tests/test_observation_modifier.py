import numpy as np
import enum
import pytest
import unittest
import json
from agent.modifiers.observation_modifier import ObservationModifier
from pysc2.tests import dummy_observation
from pysc2.lib import actions
from environment.environment import Environment


_NO_OP = actions.FUNCTIONS.no_op.id
_NO_OP_STEP = [actions.FunctionCall(_NO_OP, [])]
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

class ObservationModifierTest(unittest.TestCase):

    def setUp(self):
        json_data = '{"observations": {"screen_features": ["height_map", "player_id", "player_relative", "unit_type"], ' \
                    '"minimap_features": ["player_id", "selected"], "nonspatial_features": ["player", "score_cumulative"], ' \
                    '"action_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}, "rewards": [1, 1, 1, 1]}'
        config = json.loads(json_data)
        self._obs_mod = ObservationModifier(config["observations"], 32)
        #self.old_obs = [None]
        #self._obs_spec = {}
        #self._builder = dummy_observation.Builder(self._obs_spec)
        #self.obs = self._builder.build()
        self.env = Environment()
        #self.obs = observations = [None] * 16
        self.obs = self.env.reset()

    #def testModify(self):
    #    alt_obs = self._obs_mod.modify(self.obs[0], 0, self.old_obs[0])
    #    self._old_obs = self.obs
    #    self.obs = self.env.reset()
    #    alt_obs_2 = self._obs_mod.modify(self.obs[0], 0, alt_obs)
    #    self.assert(alt_obs[0], alt_obs_2[0])

    def testModifyScreen(self):
        print("Testing Screen Observations")
        alt_obs = self._obs_mod.modify(self.obs[0])
        self.assertIn("screen_features", alt_obs)

    def testModifyMinimap(self):
        print("Testing Minimap Observations")
        alt_obs = self._obs_mod.modify(self.obs[0])
        self.assertIn("minimap_features", alt_obs)

    def testModifyNonspatial(self):
        print("Testing Nonspatial Observations")
        alt_obs = self._obs_mod.modify(self.obs[0])
        self.assertIn("nonspatial_features", alt_obs)

    def testModifyAvailableMask(self):
        print("Testing Available Mask Observations")
        alt_obs = self._obs_mod.modify(self.obs[0])
        self.assertIn("available_mask", alt_obs)

    def testModifyAvailableActions(self):
        print("Testing Available Actions Observations")
        alt_obs = self._obs_mod.modify(self.obs[0])
        self.assertNotIn("available_actions", alt_obs)



