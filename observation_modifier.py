# #for rewards
# units killed
# enemy units killed
# damage taken per unit
# damage taken total
import numpy as np
import enum
#from reward_modifier import RewardModifier

#TODO: need an array of dictionaries
# each dictionary is for one action
# the keys: screen, minimap, nonspatial, available_actions

# json file: kinda like a dictionary but not python-related
# keys and values nested

ScreenDict = {
    "height_map": 0,
    "visibility_map": 1,
    "creep": 2,
    "power": 3,
    "player_id": 4,
    "player_relative": 5,
    "unit_type": 6,
    "selected": 7,
    "unit_hit_points": 8,
    "unit_hit_points_ratio": 9,
    "unit_energy": 10,
    "unit_energy_ratio": 11,
    "unit_shields": 12,
    "unit_shields_ratio": 13,
    "unit_density": 14,
    "unit_density_aa": 15,
    "effects": 16
}

MinimapDict = {
    "height_map": 0,
    "visibility_map": 1,
    "creep": 2,
    "camera": 3,
    "player_id": 4,
    "player_relative": 5,
    "selected": 6
}

class ObservationModifier:
    def __init__(self, config):
        # TODO: this will be replaced by something more specific
        self.config = config #dictionary

    def modify(self, obs, reward, old_observation):
        print("observation modifier")
        obs_dictionary = {}

        if "screen_features" in self.config:
            print("parsing screen features")
            # b = obs.observation.player
            # #a = obs.observation.feature_screen.player_id
            # c = obs.observation["feature_screen"] #['player_id']
            # d = (np.array(c))
            # print(d)
            o = np.array(obs.observation.feature_screen)[ScreenDict["player_id"]]
            print(o)
            # for i in self.config["screen_features"]:
            #     print("o")
            obs_dictionary["screen_features"] = [(np.array(obs.observation.feature_screen))[ScreenDict(i)] for i in self.config["screen_features"]]

        if "minimap_features" in self.config:
            print("parsing minimap features")
            obs_dictionary["minimap_features"] = [obs.observation.feature_minimap[i] for i in self.config["minimap_features"]]

        if "nonspatial_features" in self.config:
            print("parsing nonspatial features")
            obs_dictionary["nonspatial_features"] = [obs.observation[i] for i in self.config["nonspatial_features"]]

        if "action_ids" in self.config:
            print("parsing action ids")
            action_ids = list()
            for i in range(len(self.config["action_ids"])):
                if self.config["action_ids"][i] in obs.observation.available_actions:
                    action_ids.append(i)

            obs_dictionary["action_ids"] = action_ids

        print(obs_dictionary["action_ids"])
        print(obs_dictionary["nonspatial_features"][1])

        return obs_dictionary


#   old format, remove later (here for reference)

#pull out id's from config file

# #action ids
# _NO_OP = 0
# _MOV_CAM = 1
# _SELECT_POINT = 2
# _SELECT_RECT = 3
# _SELECT_CONTROL = 4
# _SELECT_UNIT = 5
# _SELECT_ARMY = 7
# _ATTACK_SCREEN = 12
# _ATTACK_MINIMAP = 13
# _STOP_QUICK = 453
# _SMART_SCREEN = 451
# _SMART_MINIMAP = 452
# _MOVE_SCREEN = 331
# _MOVE_MINIMAP = 332
# _HOLD_POS_QUICK = 274
# ################################################################
#
#     def intersection(self, lst1, lst2):
#         lst3 = [value for value in lst1 if value in lst2]
#         return lst3
#
#     def modify(self, obs, reward, old_observation):
#         feature_units = np.array(obs.observation.feature_units)
#         print("total value units: ", obs.observation.score_cumulative[3])
#         print("killed value units: ", obs.observation.score_cumulative[5])  # what does this value mean?
#
#         enemy_units = None
#         friendly_units = None
#
#         for i in range(len(feature_units)):  ##for all visible units, record info
#             unit = np.zeros(4)
#             unit[0] = feature_units[i][2]  # health
#             unit[1] = feature_units[i][7]  # health ratio
#             unit[2] = feature_units[i][12]  # x
#             unit[3] = feature_units[i][13]  # y
#
#             if (feature_units[i][11] == 1):  # friendly unit #11 = "owner"
#                 if friendly_units is None:
#                     friendly_units = [unit]
#                 else:
#                     friendly_units = np.append(friendly_units, [unit], axis=0)
#             else:  # enemy unit
#                 if enemy_units is None:
#                     enemy_units = [unit]
#                 else:
#                     enemy_units = np.append(enemy_units, [unit], axis=0)
#
#         # TODO: modify available actions
#         actions = obs.observation.available_actions
#         print("Actions: ", actions)
#
#         # 0 no_op
#         # 1 move_cam
#         # 2 select_point
#         # 3 select_rect
#         # 5 select_unit
#         # 7 select_army ?
#         # 12 attack_screen
#         # 13 attack_minimap
#         # 453 stop_quick
#         # 451 smart_screen
#         # 452 smart_minimap ?
#         # 331 move_screen
#         # 332 move_minimap
#
#         wantedActions = [_MOV_CAM, _SELECT_POINT, _SELECT_RECT, _SELECT_UNIT, _SELECT_ARMY, _ATTACK_MINIMAP,
#                          _STOP_QUICK, _SMART_SCREEN, _SMART_MINIMAP, _MOVE_SCREEN, _MOVE_MINIMAP]
#
#         actions = self.intersection(actions, wantedActions)
#
#         army_count = obs.observation['player'][8]
#         new_observation = [friendly_units, enemy_units, army_count, actions]
#         # print("TESTING!")
#         # print(np.array(obs.observation.feature_minimap))
#
#         if old_observation is not None:
#             if (new_observation[2] < old_observation[2]):
#                 print("Lost ", old_observation[2] - new_observation[2], " zerglings :(")
#
#         # TODO: All unit coordinates from minimap (if possible figure out how to split between enemy and friendly)
#             # used to determine how many enemy units vs friendly units are on the screen
#
#         return new_observation
