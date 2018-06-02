import tensorflow as tf
from runner import A2CRunner
from agent import A2CAgent
from agent import RandomAgent
from environment import Environment


def main():
    print("Commencing magic...")
    sess = tf.Session()

    env = Environment()
    agent = A2CAgent()
    #agent = RandomAgent()
    runner = A2CRunner(agent, env)
    runner.learn()


if __name__ == "__main__":
    main()
