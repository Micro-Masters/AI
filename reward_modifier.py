class RewardModifier:
    def __init__(self, config):
        # TODO: this will be replaced by something more specific
        self.config = config

    def modify(self, observation, reward, last_observation):
        return reward
