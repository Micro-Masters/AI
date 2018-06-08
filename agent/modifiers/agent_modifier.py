from agent.modifiers.observation_modifier import ObservationModifier
from agent.modifiers.reward_modifier import RewardModifier
from pysc2.lib import actions

class AgentModifier:
    def __init__(self, config, resolution_size):
        self.observation_modifier = ObservationModifier(config['observations'], resolution_size)
        self.resolution_size = resolution_size
        self.reward_modifier = RewardModifier(config['rewards'])
        self.action_ids = config['observations']['actions_ids']
        self.num_actions = len(self.actions_ids)
        self.feature_names = [config['observations']['screen_features'],
                              config['observations']['minimap_features'],
                              config['observations']['nonspatial_features']]
        self.observation_shapes = self.observation_modifier.shapes
        self.observation_data_format = self.observation_modifier.data_format

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

    # Takes an array of policy actions and returns the respective PySC2 actions
    def make_action(self, policy_actions):
        policy_fns, policy_args = policy_actions
        pysc2_actions = []
        for i in range(policy_fns.shape[0]):
            action_id = self.action_ids[policy_fns[i]]
            action_args = []
            for arg_type in actions.FUNCTIONS[action_id].args:
                arg_val = policy_args[arg_type][i]
                if self._is_spatial_arg(arg_type):
                    action_arg = [arg_val % self.resolution_size, arg_val // self.resolution_size]
                else:
                    action_arg = [arg_val]
                action_args.append(action_arg)
            pysc2_actions.append(actions.FunctionCall(action_id, action_args))

    def _is_spatial_arg(self, name):
        return name == 'screen' or name == 'screen2' or name == 'minimap'
