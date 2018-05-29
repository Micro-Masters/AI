#be sure to sample actions either here or in agent (use just one from softmax probs, not all)
import numpy as np
from keras.layers import Dense
from keras.models import Sequential
from keras.optimizers import Adam

class Policy:
    def __init__(self, state_size, action_size):
        #self.actor = build_actor()
        #self.critic = build_critic()
        self.action_size = action_size
        self.state_size = state_size

        #def build_actor(self):
        actor = Sequential()
        print(type(state_size))
        actor.add(Dense(24, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
        actor.add(Dense(action_size, activation='softmax', kernel_initializer='he_uniform'))
        actor.summary()
        actor.compile(loss="categorical_crossentropy", optimizer=Adam(lr=.001))
        #return actor

        #def build_critic(self):
        critic = Sequential()
        critic.add(Dense(24, input_dim=state_size, activation='relu', kernel_initializer='he_uniform'))
        critic.add(Dense(1, activation='linear', kernel_initializer='he_uniform'))
        critic.summary()
        critic.compile(loss="mse", optimizer=Adam(lr=.005))
        #return critic
        self.actor = actor
        self.critic = critic

    def getAction(self, observation):
        policy = self.actor.predict(observation, batch_size=1).flatten()
        return np.random.choice(self.action_size, 1, p=policy)[0]


class FullyConvLSTM:
    def __init__(self):
        print('A2C Policy init')

    def build(self):
        print('Build FullyConvLSTM')



