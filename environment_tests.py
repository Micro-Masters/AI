from pysc2.lib import actions
import time

_NO_OP = actions.FUNCTIONS.no_op.id
_NO_OP_STEP = [actions.FunctionCall(_NO_OP, [])]
_SELECT_ARMY = actions.FUNCTIONS.select_army.id
_ATTACK_MINIMAP = actions.FUNCTIONS.Attack_minimap.id
_NOT_QUEUED = [0]
_QUEUED = [1]
_SELECT_ALL = [2]


"""
this test is just to make sure that the environment is set up correctly, 
without the need to test through the runner or the agent.
"""
def test_env(env):
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