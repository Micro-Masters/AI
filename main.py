import json
import tensorflow as tf
from agent.runner import A2CRunner
from agent.agent import A2CAgent
from agent.modifiers.agent_modifier import AgentModifier
from environment.environment import Environment
from tests.environment_tests import test_env


def main():

    json_data = '{"observations": {"screen_features": ["height_map", "player_id", "player_relative", "unit_type"], ' \
                '"minimap_features": ["player_id", "selected"], "nonspatial_features": ["player", "score_cumulative"], ' \
                '"action_ids": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]}, "rewards": [1, 1, 1, 1]}'
    config = json.loads(json_data)
    print("config:")
    print(config)
    print("config.observations:")
    print(config["observations"])
    print("config.onservations.action_ids:")
    print(config["observations"]["action_ids"])
    print("config.rewards:")
    print(config["rewards"])


    print("Commencing magic...")
    sess = tf.Session()

    env = Environment()
    #test_env(env, config)

    agent_modifier = AgentModifier(config, 32)
    agent = A2CAgent(sess, agent_modifier)
    #agent = RandomAgent() ##for debugging TODO: delete later

    runner = A2CRunner(agent, env)
    runner.begin()


if __name__ == "__main__":
    main()