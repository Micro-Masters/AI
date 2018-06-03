# Copyright 2017 Google Inc. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS-IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""A base agent to write custom scripted agents."""

from pysc2.agents import base_agent

from pysc2.lib import actions
from pysc2.lib import features

import time

# Functions
_BUILD_SUPPLYDEPOT = actions.FUNCTIONS.Build_SupplyDepot_screen.id
_BUILD_BARRACKS = actions.FUNCTIONS.Build_Barracks_screen.id
_NOOP = actions.FUNCTIONS.no_op.id
_SELECT_POINT = actions.FUNCTIONS.select_point.id
_TRAIN_MARINE = actions.FUNCTIONS.Train_Marine_quick.id
_RALLY_UNITS_MINIMAP = actions.FUNCTIONS.Rally_Units_minimap.id
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id

# Features
_PLAYER_RELATIVE = features.SCREEN_FEATURES.player_relative.index
_UNIT_TYPE = features.SCREEN_FEATURES.unit_type.index

# Unit IDs
_ZERG_COMMANDCENTER = 18
_ZERG_SCV = 45
_ZERG_BARRACKS = 21

# Parameters
_PLAYER_SELF = 1
_NOT_QUEUED = [0]
_QUEUED = [1]
_SUPPLY_USED = 3
_SUPPLY_MAX = 4


class PracticeAgent(base_agent.BaseAgent):
    """A base agent to write custom scripted agents."""
    base_top_left = None
    supply_depot_built = False
    barracks_built = False
    scv_selected = False
    barracks_selected = False
    barracks_rallied = False
    army_selected = False
    army_rallied = False

    def step(self, obs):
        super(PracticeAgent, self).step(obs)

        time.sleep(0.5)

        if self.base_top_left is None:
            player_y, player_x = (obs.observation["minimap"][_PLAYER_RELATIVE] == _PLAYER_SELF).nonzero()
            self.base_top_left = player_y.mean() <= 31

        if not self.supply_depot_built:
            if not self.scv_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _ZERG_SCV).nonzero()

                target = [unit_x[0], unit_y[0]]
                self.scv_selected = True

                return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            elif _BUILD_SUPPLYDEPOT in obs.observation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _ZERG_COMMANDCENTER).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 0, int(unit_y.mean()), 20)

                self.supply_depot_build = True

                return actions.FunctionCall(_BUILD_SUPPLYDEPOT, [_NOT_QUEUED, target])
        elif not self.barracks_built:
            if _BUILD_BARRACKS in obs.obsercation["available_actions"]:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _ZERG_COMMANDCENTER).nonzero()

                target = self.transformLocation(int(unit_x.mean()), 20, int(unit_y.mean()), 0)

                self.barracks_built = True

                return actions.FunctionCall(_BUILD_BARRACKS, [_NOT_QUEUED, target])
        elif not self.barracks_rallied:
            if not self.barracks_selected:
                unit_type = obs.observation["screen"][_UNIT_TYPE]
                unit_y, unit_x = (unit_type == _ZERG_BARRACKS).nonzero()

                if unit_y.any():
                    target = [int(unit_x.mean()), int(unit_y.mean())]

                    self.barracks_selected = True

                    return actions.FunctionCall(_SELECT_POINT, [_NOT_QUEUED, target])
            else:
                self.barracks_rallied = True

                if self.base_top_left:
                    return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 21]])

                return actions.FunctionCall(_RALLY_UNITS_MINIMAP, [_NOT_QUEUED, [29, 46]])
        elif obs.observation["player"][_SUPPLY_USED] and _TRAIN_MARINE in obs.observation["available_actions"]:
            return actions.FunctionCall(_TRAIN_MARINE, [_QUEUED])
        elif not self.army_rallied:
            if not self.army_selected:
                if _SELECT_ARMY in obs.observation["available_actions"]:
                    self.army_selected = True
                    self.barracks_selected = False

                    return actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])
                elif _ATTACK_MINIMAP in obs.observation["available_actions"]:
                    self.army_rallied = True
                    self.army_selected  = False

                    if self.base_top_left:
                        return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [39, 45]])

                    return actions.FunctionCall(_ATTACK_MINIMAP, [_NOT_QUEUED, [21, 24]])

        return actions.FunctionCall(_NOOP, [])

    def transformLocation(self, x, x_distance, y, y_distance):
        if not self.base_top_left:
            return [x - x_distance, y - y_distance]

        return [x + x_distance, y + y_distance]
