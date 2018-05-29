import gym
import sc2gym.envs

class Environment:
    def __init__(self, n_envs):
        self.n_envs = n_envs
        env = gym.make("SC2Game-v0")
        env.settings['map_name'] = 'killtanks'
        env.settings['visualize'] = True
        #self.n_obs = len(env.observation_spec)
        #self.n_act = len(env.action_spec)
        self.reset = env.reset	
