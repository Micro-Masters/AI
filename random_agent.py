import numpy
from pysc2.agents import base_agent
from pysc2.lib import actions


####################################################################################
"""A random agent for starcraft."""

"""remove later, this is for environment testing purposes only"""

class RandomAgent(base_agent.BaseAgent):
  """A random agent for starcraft."""

  def step(self, obs):
    super(RandomAgent, self).step(obs)
    function_id = numpy.random.choice(obs.observation.available_actions)
    args = [[numpy.random.randint(0, size) for size in arg.sizes]
            for arg in self.action_spec.functions[function_id].args]
    return actions.FunctionCall(function_id, args)
##################################################################################
