# #for rewards
# units killed
# enemy units killed
# damage taken per unit
# damage taken total
import numpy as np
from reward_modifier import RewardModifier

class ObservationModifier:
    def __init__(self, config):
        # TODO: this will be replaced by something more specific
        self.config = config

    def modify(self, obs, reward, old_observation):
        feature_units = np.array(obs.observation.feature_units)
        print("total value units: ", obs.observation.score_cumulative[3])
        print("killed value units: ", obs.observation.score_cumulative[5]) #what does this value mean?

        enemy_units = None
        friendly_units = None

        for i in range(len(feature_units)): ##for all visible units, record info
            unit = np.zeros(4)
            unit[0] = feature_units[i][2] #health
            unit[1] = feature_units[i][7] #health ratio
            unit[2] = feature_units[i][12] #x
            unit[3] = feature_units[i][13] #y

            if(feature_units[i][11] == 1): #friendly unit #11 = "owner"
                if friendly_units is None:
                    friendly_units = [unit]
                else:
                    friendly_units = np.append(friendly_units, [unit], axis=0)
            else: #enemy unit
                if enemy_units is None:
                    enemy_units = [unit]
                else:
                    enemy_units = np.append(enemy_units, [unit], axis=0)

        # TODO: modify available actions
        actions = obs.observation.available_actions

        army_count = obs.observation['player'][8]
        new_observation = [friendly_units, enemy_units, army_count, actions]

        if old_observation is not None:
            if(new_observation[2] < old_observation[2]):
                print("Lost ", old_observation[2] - new_observation[2], " zerglings :(")

        # TODO: calculate damage taken by enemy units ("health" above is only for those that are visible?)
        # possible problem: to calculate damage, we need to compare health at last obs to health at this obs.
        # but, if some units move out of view, or the camera moves, how do we match up which units were in the previous
        # obs vs which units are in the current obs in order to make the calculations??

        # TODO: call reward modifier
        # if old_observation is not None:
        #     reward = self.reward_mod.modify(new_observation, reward, old_observation)

        return new_observation




