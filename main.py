import tensorflow as tf
from agent.runner import A2CRunner
from agent.agent import A2CAgent
from environment.environment import Environment
from tests.environment_tests import test_env


def main():
    print("Commencing magic...")
    sess = tf.Session()

    env = Environment()
    test_env(env)

    agent = A2CAgent()
    #agent = RandomAgent() ##for debugging TODO: delete later

    runner = A2CRunner(agent, env)
    runner.begin()


if __name__ == "__main__":
    main()