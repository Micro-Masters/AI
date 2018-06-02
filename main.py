import tensorflow as tf
from runner import A2CRunner
from agent import A2CAgent
from random_agent import RandomAgent
from environment import Environment

from pysc2.lib import actions
_NO_OP = actions.FUNCTIONS.no_op.id
_NO_OP_STEP = [actions.FunctionCall(_NO_OP, [])]

def main():
    print("Commencing magic...")
    sess = tf.Session()

    env = Environment()

    #agent = A2CAgent() TODO: uncomment to test agent

    ###TODO: delete this later
    ###for testing purposes only
    for x in range(100):
        env.step(_NO_OP_STEP)
    print("STEPPING ON THE BEAT WOOT WOOT WOOT WOOT")

    agent = RandomAgent() ##for debugging TODO: delete later

    runner = A2CRunner(agent, env)
    runner.learn()

if __name__ == "__main__":
    main()