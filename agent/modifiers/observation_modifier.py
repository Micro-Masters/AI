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
        # self.shapes = {
        #     'screen': [None, len(config['screen_features']), resolution_size, resolution_size],
        #     'minimap': [None, len(config['minimap_features']), resolution_size, resolution_size],
        #     'nonspatial': [None, len(config['nonspatial_features']), 11],
        #     'available_mask': [None, len(config['action_ids'])]
        # }
        self.shapes = {
            'screen': [len(config['screen_features']), resolution_size, resolution_size],
            'minimap': [len(config['minimap_features']), resolution_size, resolution_size],
            'nonspatial': [len(config['nonspatial_features']), 11],
            'available_mask': [len(config['action_ids'])]
        }
        print("HELLOOOO")
        print(self.shapes['screen'])
        self.data_format = 'NCHW'

    def modify(self, obs):
        print("observation modifier")
        obs_dictionary = {}

        if "screen_features" in self.config:
            print("parsing screen features")
            obs_dictionary["screen_features"] = [(np.array(obs.observation.feature_screen))[ScreenDict[i]] for i in self.config["screen_features"]]
            # print(type(obs_dictionary["screen_features"]))
            # assert all(x.shape == (32, 32) for x in obs_dictionary["screen_features"])

        if "minimap_features" in self.config:
            print("parsing minimap features")
            obs_dictionary["minimap_features"] = [(np.array(obs.observation.feature_minimap))[MinimapDict[i]] for i in self.config["minimap_features"]]
            # print(type(obs_dictionary["minimap_features"]))
            # assert all(x.shape == (32, 32) for x in obs_dictionary["minimap_features"])

        if "nonspatial_features" in self.config:
            print("parsing nonspatial features")
            # non_space_feat = list()
            # for i in range(len(self.config["nonspatial_features"])):
            #     non_space_feat.append(obs.observation[i])
            # obs_dictionary["nonspatial_features"] = non_space_feat

            obs_dictionary["nonspatial_features"] = [np.array(obs.observation[i]) for i in self.config["nonspatial_features"]]
            print(type(obs_dictionary["nonspatial_features"]))

        #TODO: shouldn't be a list, should be array of lists??
        if "action_ids" in self.config:
            print("parsing action ids")
            action_mask = list()
            for i in range(len(self.config["action_ids"])):
                if self.config["action_ids"][i] in obs.observation.available_actions:
                    action_mask.append(1)
                else:
                    action_mask.append(0)

            #obs_dictionary["available_mask"] = action_mask
            obs_dictionary["available_mask"] = np.array(action_mask, dtype=np.int32) #XXX
            print(type(obs_dictionary["available_mask"]))
            #print(obs_dictionary["available_mask"][0].shape)

        # print("MODIFIER:")
        # print(obs_dictionary[0])
        # print(obs_dictionary[1])
        # print(obs_dictionary[2])
        # print(obs_dictionary[3])

        return obs_dictionary

