import tensorflow as tf
import numpy as np

class A2CRunner:
    def __init__(self, agent, env, n_updates=10000, n_steps=16, train=True):
        print('A2C Runner init')
        self.agent = agent
        self.n_updates = n_updates
        self.n_steps = n_steps
        self.observation = None

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

        for i in range(self.n_steps):
            action, value = self.agent.act(self.observation)
            actions[i], values[i] = action, value
            observations[i] = self.observation
            self.observation, rewards[i], dones[i] = self.env.step(action)

        return observations, actions, rewards, values, dones
