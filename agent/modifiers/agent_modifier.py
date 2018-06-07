from agent.modifiers.observation_modifier import ObservationModifier
from agent.modifiers.reward_modifier import RewardModifier


class AgentModifier:
    def __init__(self, config):
        self.observation_modifier = ObservationModifier(config['observations'])
        self.reward_modifier = RewardModifier(config['rewards'])
        self.observation_shapes = self.observation_modifier.shapes
        self.actions_ids = config['observations']['actions']
        self.num_actions = len(self.actions_ids)

    # Modifies the rewards, takes in arrays of n_envs length (not known)
    def modify_reward(self, observations, rewards, last_observations):
        return [self.reward_modifier.modify(obs, rwd, last_obs)
                for obs in observations
                for rwd in rewards
                for last_obs in last_observations]

    # Modifies the observations, takes in arrays of n_envs length (not known)
    # Return an array of dictionaries with keys: screen, minimap, nonspatial, available_actions
    def modify_observation(self, observations):
        return [self.observation_modifier.modify(obs) for obs in observations]

    # Takes an action number (index in self.actions) and returns the respective PySC2 action
    def make_action(self, policy_action):
        #TODO take out from policy and make into PySC2 action
        print('Make action')