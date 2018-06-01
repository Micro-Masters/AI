from multiprocessing import Process, Pipe
from pysc2.env import sc2_env
from functools import partial

#references:
#   https://github.com/openai/baselines/blob/master/baselines/common/vec_env/subproc_vec_env.py
#   https://github.com/simonmeister/pysc2-rl-agents/blob/master/rl/environment.py


class Environment:
    def __init__(self, n_envs=1):
        self.n_envs = n_envs

        ##set up arguments to give to sc2_env upon creation
        size_px = (32, 32)
        env_args = dict(
            map_name='zergmap', ##TODO: set map name
            step_mul=8, ##number of game steps per agent step
            game_steps_per_episode=0,
            screen_size_px=size_px, ##screen and minimap resolution
            minimap_size_px=size_px)

        ##add visualization if running only one game environment
        #TODO: allow user to specify number of game instances visualized
        if(n_envs == 1):
            env_args['visualize'] = 'store_true'

        ##using args, create callable functions for game initialization for each worker in pipe
        env_fns = [partial(make_sc2env, **env_args)] * n_envs

        ##create pipe of workers
        self.remotes, self.work_remotes = zip(*[Pipe() for _ in range(n_envs)])
        self.ps = [Process(target=worker, args=(work_remote, CloudpickleWrapper(env_fn)))
                   for (work_remote, env_fn) in zip(self.work_remotes, env_fns)]
        for p in self.ps:
            p.start()

    #def launch(self): ##Add bak in later??
    #    for i in range(self.n_envs):

    ##send action to each worker
    def step(self, actions):
        print('env step')
        for remote, action in zip(self.remotes, actions):
            remote.send(("step", action))
        timesteps = [remote.recv() for remote in self.remotes]
        return timesteps

    ##reset each worker
    def reset(self):
        print('reset')
        for remote, action in zip(self.remotes, actions):
            remote.send(("reset", [None]*self.n_envs))
        timesteps = [remote.recv() for remote in self.remotes]
        return timesteps

    ##close the pipe
    def close(self):
        for remote in self.remotes:
            remote.send(('close', None))
        for p in self.ps:
            p.join()

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
    env = sc2_env.SC2Env(**kwargs)
    return env
