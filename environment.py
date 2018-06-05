from multiprocessing import Process, Pipe
from pysc2.env import sc2_env
from pysc2.env.sc2_env import Agent, Bot
from pysc2.lib import features
from pysc2 import maps
from pysc2.maps import lib
from functools import partial
from absl import flags
FLAGS = flags.FLAGS
FLAGS(['environment.py']) #XXX

_TERRAN = 2
_ZERG = 3

#references:
#   https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
#   https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/environment.py
#   https://github.com/deepmind/pysc2/blob/master/pysc2/env/sc2_env.py
#       other pysc2 git repos too, but above is the main one

class Environment:
    def __init__(self, n_envs=1): #n_envs = 1
        self.n_envs = n_envs

        env_args = self.getArgs()

        #using args, create callable functions for game initialization for each worker in pipe
        env_fns = [partial(make_sc2env, **env_args)] * n_envs

        #create pipe of workers
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

    def launch(self): ##Add bak in later??
        for i in range(self.n_envs):
            print('i = ', i)

    def step(self, actions):
        """send action to each worker"""
        print('env step')
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        timesteps = [remote.recv() for remote in self.remotes]
        return timesteps

    def reset(self):
        """reset each worker"""
        print('reset')
        actions = [None] * self.n_envs
        for remote, action in zip(self.remotes, actions):
            remote.send(("reset", [None]*self.n_envs))
        timesteps = [remote.recv() for remote in self.remotes]
        return timesteps

    def close(self):
        """close the pipe"""
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

    def getArgs(self):
        """
        set up arguments to give to sc2_env upon creation
        """
        agent_type = Agent(_ZERG) #specify agent race
        bot_type = Bot(_TERRAN, 1) #specify bot race and difficulty
        player_types = [agent_type, bot_type]

        size_px = (32, 32)
        dimensions = features.Dimensions(screen=size_px, minimap=size_px) ##TODO
        #note: screen size must be at least as big as minimap

        agent_interface = features.AgentInterfaceFormat(feature_dimensions=dimensions,
                                                        use_feature_units=True) ##check exactly what feature units are

        env_args = dict(
            map_name="Zerg_44_36", ##TODO: set map name to something else
            step_mul=8, ##number of game steps per agent step
            game_steps_per_episode=0,
            score_index=-1, #uses win/loss reward. change later to use map default
            disable_fog=True,
            agent_interface_format=agent_interface,
            players=player_types)
            #screen_size_px=size_px, ##depr
            #minimap_size_px=size_px)###depr

        #add visualization if running only one game environment

        if(self.n_envs == 1):
        #if(self.n_envs != 0):
            env_args['visualize'] = True

        # TODO: allow user to specify number of game instances visualized
        return env_args

    ##TODO: delete later?
    def get_action_specs(self):
        actions = [None] * self.n_envs
        for remote, action in zip(self.remotes, actions):
            remote.send(("action_spec", [None]*self.n_envs))
        specs = [remote.recv() for remote in self.remotes]
        return specs

class Zerg_44_36(lib.Map):
    """
    subclassing of map type. needed for map to be recognised by pysc2
    this map must be in Starcraft Map directory
    """
    prefix = "" # "/home/kirsten/StarCraftII/ECS170Project/AI/"
    filename = "Zerg_44_36"
    players = 2

def worker(remote, env_fn_wrapper):
    """
    A worker is like an instance of the game.
    """
    env = env_fn_wrapper.x()
    while True:
        cmd, action = remote.recv()
        if cmd == 'step':
            timesteps = env.step([action])
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == 'reset':
            timesteps = env.reset()
            assert len(timesteps) == 1
            remote.send(timesteps[0])
        elif cmd == 'close':
            remote.close()
            break
        elif cmd == 'observation_spec':
            spec = env.observation_spec()
            remote.send(spec)
        elif cmd == 'action_spec':
            spec = env.action_spec()
            remote.send(spec)
        else:
            raise NotImplementedError

class CloudpickleWrapper(object):
    """
    Uses cloudpickle to serialize contents (otherwise multiprocessing tries
    to use pickle).
    """
    def __init__(self, x):
        self.x = x

    def __getstate__(self):
        import cloudpickle
        return cloudpickle.dumps(self.x)

    def __setstate__(self, ob):
        import pickle
        self.x = pickle.loads(ob)


def make_sc2env(**kwargs):
    """
    function called by each environment when created in pipe
    when called, **kwargs comes from getArgs() function
    """
    env = sc2_env.SC2Env(**kwargs)
    return env