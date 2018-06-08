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
    def __init__(self, config, resolution_size):
        self.config = config #dictionary
        self.shapes = {
            'screen': [len(config['screen_features']), resolution_size, resolution_size],
            'minimap': [len(config['minimap_features']), resolution_size, resolution_size],
            'nonspatial': [len(config['nonspatial_features']), 11],
            'available_mask': [len(config['action_ids'])]
        }
        self.data_format = 'NCHW'

    def modify(self, obs):
        print("observation modifier")
        obs_dictionary = {}

        if "screen_features" in self.config:
            print("parsing screen features")
            obs_dictionary["screen_features"] = [(np.array(obs.observation.feature_screen))[ScreenDict[i]] for i in self.config["screen_features"]]

        if "minimap_features" in self.config:
            print("parsing minimap features")
            obs_dictionary["minimap_features"] = [(np.array(obs.observation.feature_minimap))[MinimapDict[i]] for i in self.config["minimap_features"]]

        if "nonspatial_features" in self.config:
            print("parsing nonspatial features")
            obs_dictionary["nonspatial_features"] = [obs.observation[i] for i in self.config["nonspatial_features"]]

        if "action_ids" in self.config:
            print("parsing action ids")
            action_mask = list()
            for i in range(len(self.config["action_ids"])):
                if self.config["action_ids"][i] in obs.observation.available_actions:
                    action_mask.append(1)
                else:
                    action_mask.append(0)

            obs_dictionary["available_mask"] = action_mask

        return obs_dictionary

