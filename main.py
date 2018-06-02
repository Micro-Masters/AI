import tensorflow as tf
import time
from runner import A2CRunner
from agent import A2CAgent
from random_agent import RandomAgent
from environment import Environment
from pysc2.env import sc2_env
from pysc2.lib import actions

_NO_OP = actions.FUNCTIONS.no_op.id
_NO_OP_STEP = [actions.FunctionCall(_NO_OP, [])]
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]

def main():
    print("Commencing magic...")
    sess = tf.Session()

    env = Environment()
    test(env)

    agent = A2CAgent()
    #agent = RandomAgent() ##for debugging TODO: delete later

    runner = A2CRunner(agent, env)
    runner.begin()

def test(env):
    ###TODO: delete this later
    ###for testing purposes only
    obs = env.reset()
    for x in range(2):
        print("no op")
        env.step(_NO_OP_STEP)
        print("select_army")
        env.step([actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])])
        print("attacking")
        env.step([actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [20, 20]])])
        print("no op")
        for i in range(1000):
            if(1%10 == 1):
                time.sleep(1)
            env.step(_NO_OP_STEP)

if __name__ == "__main__":
    main()