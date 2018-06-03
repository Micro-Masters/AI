from pysc2.lib import actions
import time
import numpy as np

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
    counts = np.zeros(1000)

    obs = env.reset()
    for x in range(2):
        print("no op")
        temp = env.step(_NO_OP_STEP)
        print("select_army")
        env.step([actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])])
        print("attacking")
        env.step([actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [20, 20]])])
        print("no op")
        for i in range(1000):
            check_obs(obs, i, counts)
            if(1%10 == 1):
                time.sleep(1)
            obs = env.step(_NO_OP_STEP)


def check_obs(obs, i, counts):

    print("army count: ", obs[0].observation['player'][8])
    counts[i] = obs[0].observation['player'][8]

    print("score_cumulative[total_value_units]: ", obs[0].observation.score_cumulative[3])
    print("score_cumulative[killed_value_units]: ", obs[0].observation.score_cumulative[5])


    np.save('feature_unit_file', obs[0].observation.feature_units)  # save the file as "outfile_name.npy"

    temp = obs[0].observation.feature_units

    print("feature_unit test: ", np.array(temp))
