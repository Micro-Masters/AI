import numpy as np


class A2CRunner:
    def __init__(self, agent, env, n_updates=10000, n_steps=16, train=True):
        self.agent = agent
        self.env = env
        self.n_updates = n_updates
        self.n_steps = n_steps
        self.observation = None

    # Begin running and learning (if desired)!
    def begin(self):
        # Run for n_updates batches, training if self.train=True
        for i in range(self.n_updates):
            train_args = self.run_batch()
            if self.train:
                self.agent.train(*train_args)

    # Run a batch and return the results
    def run_batch(self):
        # Define 2D information arrays of size n_steps by n_envs and type float
        shapes = (self.n_steps, self.env.n_envs)
        rewards = np.zeros(shapes, dtype=np.float32)
        values = np.zeros(shapes, dtype=np.float32)
        dones = np.zeros(shapes, dtype=np.float32)
        # We will assign observation arrays of n_envs length so these become multidimensional too
        observations = [None] * self.n_steps
        actions = [None] * self.n_steps

        # For every step, get an action to perform and then perform it (recording results)
        for i in range(self.n_steps):
            action, value = self.agent.act(self.observation)
            actions[i], values[i] = action, value
            observations[i] = self.observation
            self.observation, rewards[i], dones[i] = self.env.step(action)

            # Modify the reward given such a modifier
            rewards[i] = self.agent.agent_modifier.modify_reward(
                self.observation, rewards[i], observations[i])

        # Get the next value (for use in returns calculation)
        next_value = self.agent.act(self.observation)

        return observations, actions, rewards, dones, values, next_value


