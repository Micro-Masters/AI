from multiprocessing import Process, Pipe
from pysc2.env import sc2_env


class Environment:
    def __init__(self, env_args, n_envs=1):
        self.env_args = env_args
        self.n_envs = n_envs

    def launch(self):
        for i in range(self.n_envs):


    def step(self, action):
        print('env step')