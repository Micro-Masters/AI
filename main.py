import tensorflow as tf
import sys
import numpy as np
from runner import A2CRunner
from agent import A2CAgent
from environment import Environment
from policies import Policy #FullyConvLSTM
from absl import flags
FLAGS = flags.FLAGS
###

def main():
    print("Commencing magic...")
    FLAGS(sys.argv)
    sess = tf.Session()

    env = Environment(1)
    state_size = 8 #env.n_obs #env.observation_spec #space.shape[0]
    action_size = 5 #actor.getActionLen#len(smart_actions)
    #action_size = env.n_act #action_spec #space.n

    policy = Policy(state_size, action_size)  
    agent = A2CAgent(sess, policy)
  
    runner = A2CRunner(agent, env, state_size, action_size)
    runner.learn()

if __name__ == "__main__":
    main()
