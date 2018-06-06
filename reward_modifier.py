from pysc2.lib import actions
import numpy as np

_MOVE_CAMERA = actions.FUNCTIONS.move_camera.id
_NO_VISIBLE_ENEMY_UNITS_PENALTY = -20 # TODO: change this to be part of config??

class RewardModifier:
    def __init__(self, config):
        # TODO: this will be replaced by something more specific
        self.config = config
        # penalize action sequences that lead to death of zerlings
        # reward action sequences that lead to damaging tanks
        # reward action sequences that keep camera centered on units ?? for more precise microing?

    '''get sub-components of our personalized reward and weight them'''
    def modify(self, new_observation, map_reward, old_observation):
        #zerglings_killed = old_observation[2] - obs.observation['player'][8]
        print("reward modifier")

        if old_observation is None:
            return 0

        '''find out if the camera has been moved'''
        if self.get_cam_action(new_observation) is True:
            cam_action = True
        else:
            cam_action = False

        '''calculate change in enemy health to get health reward'''
        # TODO: move below to different function?
        avg_enemy_health = self.get_avg_enemy_health(new_observation)
        old_avg_enemy_health = self.get_avg_enemy_health(old_observation)
        diff_health = old_avg_enemy_health - avg_enemy_health

        if old_avg_enemy_health is 0:   # camera not on enemies in last obs
            health_reward = 0
        elif avg_enemy_health is 0:     # camera not on enemies in this obs
            health_reward = 0
            camera_penalty = _NO_VISIBLE_ENEMY_UNITS_PENALTY
        elif diff_health is 0:          # no change in health
            health_reward = 0
        elif diff_health > 0:           # reward damage done
            health_reward = diff_health
        else:                           # this means more enemy units came into view - we don't want to penalize this
            health_reward = 0           # OR it means we killed unit so the average went up - probably don't want to penalize this either.

        '''penalize zergling death'''
        zergling_loss = self.get_zergling_loss(new_observation, old_observation)


        '''reward enemy kills'''
        enemy_kills = self.get_enemy_kills(new_observation, old_observation, cam_action)

        '''calculate and return modified reward'''
        reward = self.calculate_reward(map_reward, health_reward, zergling_loss, enemy_kills)

        print("reward: ", reward)
        return reward

    '''modify our calculations and the map reward by the values given by config'''
    '''this gives some components of the reward more weight than other components'''
    def calculate_reward(self, map_reward, health_reward, zergling_loss, enemy_kills):
        map_reward = map_reward * self.config[0]
        health_reward = health_reward * self.config[1]
        zergling_loss = zergling_loss * self.config[2]
        enemy_kills = enemy_kills * self.config[3]

        return map_reward + health_reward + zergling_loss + enemy_kills

    '''see if the camera has been moved since the last observation'''
    def get_cam_action(self, obs):
        last_actions = obs.observation.last_actions
        if _MOVE_CAMERA in last_actions:
            return True
        else:
            return False

    '''gets the average health of enemies'''
    def get_avg_enemy_health(self, obs):
        feature_units = np.array(obs.observation.feature_units)
        total_enemy_health = 0
        total_enemy_units = 0

        for i in range(len(feature_units)): ##for all visible units, record info
            if(feature_units[i][11] != 1): #unit is an enemy unit
                total_enemy_health += feature_units[i][2]
                total_enemy_units += 1

        if total_enemy_units == 0:
            return 0
        else:
            return total_enemy_health / total_enemy_units #average the health

    '''checks to see if zergling have died by checking our army count'''
    def get_zergling_loss(self, new_obs, old_obs):
        old_army_count = old_obs.observation['player'][8]
        new_army_count = new_obs.observation['player'][8]

        return old_army_count - new_army_count

    '''check to see if we have killed any enemy tanks'''
    def get_enemy_kills(self, new_obs, old_obs, cam_action):
        # TODO: if we can find this from the minimap later, change it to that to make it more accurate
        # for now: check if camera hasn't moved, and if enemy unit count decreased. if yes to both: we killed one!!
        if cam_action is True:
            return 0

        new_feature_units = np.array(new_obs.observation.feature_units)
        old_feature_units = np.array(old_obs.observation.feature_units)

        new_enemy_unit_count = 0
        old_enemy_unit_count = 0

        for i in range(len(new_feature_units)): ##for all visible units, record info
            if(new_feature_units[i][11] != 1): #unit is an enemy unit
                 new_enemy_unit_count += 1

        for i in range(len(old_feature_units)): ##for all visible units, record info
            if(old_feature_units[i][11] != 1): #unit is an enemy unit
                 old_enemy_unit_count += 1

        return old_enemy_unit_count - new_enemy_unit_count

