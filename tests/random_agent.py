import numpy
from pysc2.agents import base_agent
from pysc2.lib import actions
from pysc2.env import sc2_env


####################################################################################
"""A random agent for starcraft."""

"""remove later, this is for environment testing purposes only"""

class RandomAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""
  def __init__(self, action_spec):
    self.action_spec = action_spec
    self.steps = 0
    self.reward = 0

  def step(self, obs):
    super(RandomAgent, self).step(obs)
    return actions.FunctionCall(1, [30., 30])
    #print("action type:")
    #print(type(self.action_spec))
    function_id = numpy.random.choice(obs.observation.available_actions)
    #function_id = numpy.random.choice(obs.available_actions)
    args = [[numpy.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
            #for arg in self.action_spec.functions[function_id].args]
    return actions.FunctionCall(function_id, args)
##################################################################################
