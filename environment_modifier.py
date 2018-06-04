from observation_modifier import ObservationModifier
from reward_modifier import RewardModifier


class EnvironmentModifier:
    def __init__(self, config):
        # TODO: config or below inits may be replaced by something more specific
        self.observation_modifier = ObservationModifier(config)
        self.reward_modifier = RewardModifier(config)

    # Modifies the observations and rewards, takes in arrays of n_envs length (not known)
    def modify(self, observations, rewards, last_observations):
        new_observations = [self.observation_modifier.modify(obs) for obs in observations]
        new_rewards = [self.reward_modifier.modify(obs, rwd, last_obs)
                       for obs in new_observations
                       for rwd in rewards
                       for last_obs in last_observations]
        return new_observations, new_rewards
