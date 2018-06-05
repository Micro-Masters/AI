import tensorflow as tf
from runner import A2CRunner
from agent import A2CAgent
from random_agent import RandomAgent
from environment import Environment
from environment_tests import test_env


def main():
    print("Commencing magic...")
    sess = tf.Session()

    # add user specified number of game instances
    #n_envs = input ("Number of games instances to be visualized: ")
    env = Environment()
    test_env(env)

    agent = A2CAgent()
    #agent = RandomAgent() ##for debugging TODO: delete later

    runner = A2CRunner(agent, env)
    runner.begin()


if __name__ == "__main__":
    main()