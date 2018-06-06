from pysc2.lib import actions
import time
import numpy as np
from observation_modifier import ObservationModifier
from reward_modifier import RewardModifier
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

    obs_config = {}
    obs_config["action_ids"] = [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
    #obs_config["action_ids"] = [1, 0, 11, 12, 13, 14, 15, 16, 17, 18]
    obs_config["screen_features"] = ["player_id", "player_relative", "unit_type"]
    obs_config["minimap_features"] = ["player_id", "selected"]
    obs_config["nonspatial_features"] = ["player", "score_cumulative"]


    obs_mod = ObservationModifier(obs_config)
    reward_mod = RewardModifier([1, 1, 1, 1])

    old_obs = [None]
    obs = env.reset()
    alt_obs = obs_mod.modify(obs[0], 0, old_obs[0])
    reward = reward_mod.modify(obs[0], 0, old_obs[0])

    # print("friendly_units[0]: ")
    # print("(health, health ratio, x, y)")
    # print(alt_obs[0][0])
    # print("enemy_units: ")
    # print(alt_obs[1])
    # print("army_count: ")
    # print(alt_obs[2])
    # print("available_actions: ")
    # print(alt_obs[3])

    for x in range(2):
        print("no op")
        old_obs = obs
        obs = env.step(_NO_OP_STEP)
        alt_obs = obs_mod.modify(obs[0], None, alt_obs)
        reward = reward_mod.modify(obs[0], 0, old_obs[0])
        print("select_army")
        old_obs = obs
        obs = env.step([actions.FunctionCall(_SELECT_ARMY, [_NOT_QUEUED])])
        alt_obs = obs_mod.modify(obs[0], None, alt_obs)
        reward = reward_mod.modify(obs[0], 0, old_obs[0])
        print("attacking")
        old_obs = obs
        obs = env.step([actions.FunctionCall(_ATTACK_MINIMAP, [_QUEUED, [20, 20]])])
        #print("minimap " + np.array(obs[0].observation.feature_minimap['player_relative']))
        alt_obs = obs_mod.modify(obs[0], None, alt_obs)
        reward = reward_mod.modify(obs[0], 0, old_obs[0])
        print("no op")
        for i in range(1000):
            #check_obs(obs, i, counts)
            if(1%10 == 1):
                time.sleep(1)
            old_obs = obs
            obs = env.step(_NO_OP_STEP)
            alt_obs = obs_mod.modify(obs[0], None, alt_obs)
            reward = reward_mod.modify(obs[0], 0, old_obs[0])


def check_obs(obs, i, counts):

    print("army count: ", obs[0].observation['player'][8])
    counts[i] = obs[0].observation['player'][8]

    # print("score_cumulative[total_value_units]: ", obs[0].observation.score_cumulative[3])
    # print("score_cumulative[killed_value_units]: ", obs[0].observation.score_cumulative[5])

    # print("player: ", obs[0].observation['player'])
    # print(type(obs[0].observation))
    # print(type(obs[0]))
    # print(type(obs[0].observation['feature_screen']))
    # #print(type(obs[0].observation['feature_screen'][0]))
    # test = obs[0].observation['feature_screen']
    # print(test[0][0])
    # #print("obs: ", obs[0].observation)

    if(counts[i] < counts[i-1]):
        print("LOST ", counts[i] - counts[i - 1], " ZERGLINGS :( ")
    # print("test: ", type(obs[0].observation["ScreenFeatures"]))
    # print("test: ", type(obs[0].observation['feature_screen']))
    # print("test: ", (obs[0].observation['feature_screen'][1] == 3).nonzero())
    #print(obs[0].observation)
    # print("test: ", type(obs[0].observation.feature_units))
    # print("test: ", obs[0].observation.feature_units)
    # print("test: ", type(obs[0].observation["score_cumulative"]))
    # print("test: ", type(obs[0].observation["score_cumulative"][0]))
    # print("test: ", type(obs[0].observation.feature_units[0][0]))
    #print("test: ", obs[0].observation["feature_units"])

    # np.save('feature_unit_file', obs[0].observation.feature_units)  # save the file as "outfile_name.npy"

    temp = np.array(obs[0].observation.feature_units)

    print(len(temp))


    #
    # print("feature_unit test: ", np.array(temp))
