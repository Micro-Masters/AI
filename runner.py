import tensorflow as tf
import numpy as np

class A2CRunner:
    def __init__(self, agent, env, state_size, action_size, n_updates=10000, n_steps=16, train=True):
        print('A2C Runner init')
        self.agent = agent
        self.env = env
        self.n_updates = n_updates
        self.n_steps = n_steps
        self.observation = None
        self.action_size = action_size
        self.state_size = state_size

    def learn(self):
        print("learn")
        for i in range(self.n_updates):
            self.run_batch()
            self.agent.train()

    # Run a batch and return the information
    def run_batch(self):
        shapes = (self.n_steps, self.env.n_envs)
        values = np.zeros(shapes, dtype=np.float32)
        rewards = np.zeros(shapes, dtype=np.float32)
        dones = np.zeros(shapes, dtype=np.float32)
        observations = [None] * self.n_steps
        actions = [None] * self.n_steps
        self.observation = self.env.reset() #XXX
        self.t_observation = self.agent.transformObservation(self.observation)
	#TODO: transform observation
        #observation = np.reshape(observation, [1, self.state_size]) #XXX

        for i in range(self.n_steps):
            action, value = self.agent.act(self.t_observation)
            actions[i], values[i] = action, value
            observations[i] = self.observation
            self.observation, rewards[i], dones[i] = self.env.step(action)
            self.t_observation = self.agent.transformObservation(self.observation)
            #TODO: transform observation
        return observations, actions, rewards, values, dones
