# Copyright 2017-2018 MIT
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from karel_env.karel_gym_env import KarelGymEnv
from ..core.rl import *
from .karel import *
from ..core.dt import *
from ..util.log import *
#from .custom_dqn import DQN
from .ppo import PPO
from collections import Iterable
import random
from itertools import product
import gym
from gym.spaces import Box
import sys

class KarelEnvWrapper(gym.Wrapper):
    def __init__(self, env=None, op=[2, 0, 1]):
        """
        Transpose observation space for images
        """
        gym.Wrapper.__init__(self, env)
        assert len(op) == 3, "Error: Operation, " + str(op) + ", must be dim3"
        self.op = op
        obs_shape = self.observation_space.shape
        if len(obs_shape) == 3:
            self.observation_space = Box(
                self.observation_space.low[0, 0, 0],
                self.observation_space.high[0, 0, 0], [
                    obs_shape[self.op[0]], obs_shape[self.op[1]],
                    obs_shape[self.op[2]]
                ],
                dtype=self.observation_space.dtype)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        perception_vector = self.env._world.get_perception_vector()
        return (self.observation(ob.astype(np.float32)), np.array(perception_vector, dtype=np.float32)), float(reward), done, {}

    def reset(self):
        ob = self.observation(np.array(self.env.reset(), dtype=np.float32))
        perception_vector = np.array(self.env._world.get_perception_vector(), np.float32)
        return ob, perception_vector

    def observation(self, ob):
        if len(self.observation_space.shape) == 3:
            return np.transpose(ob, (self.op[0], self.op[1], self.op[2]))
        return ob

environments = [
                'cleanHouse',
                'fourCorners',
                'harvester',
                'randomMaze',
                'stairClimber_sparse',
                'topOff',
                ]
env_to_hw = dict(
    cleanHouse=(14, 22),
    fourCorners=(12, 12),
    harvester=(8, 8),
    #randomMaze=(8, 8),
    randomMaze=(100, 100),
    #stairClimber_sparse=(12, 12),
    stairClimber_sparse=(100, 100),
    topOff=(12, 12),
)
print("STAIRCLIMBER 100x100")
env_to_time = dict(
    cleanHouse=300,
    fourCorners=100,
    harvester=100,
    randomMaze=300,
    stairClimber_sparse=100,
    topOff=100,
)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def _generate_run_name(parameters,
                       id,
                       repeat,
                       token_len=20,
                       max_len=255):
    """Generate a run name by writing abbr parameter key-value pairs in it,
    for an easy curve comparison between different search runs without going
    into the gin texts.

    Args:
        parameters (dict): a dictionary of parameter configurations
        id (int): an integer id of the run
        repeat (int): an integer id of the repeats of the run
        token_len (int): truncate each token for so many chars
        max_len (int): the maximal length of the generated name; make sure
            that this value won't exceed the max allowed filename length in
            the OS

    Returns:
        str: a string with parameters abbr encoded
    """

    def _abbr_single(x):
        def _initials(t):
            words = [w for w in t.split('_') if w]
            len_per_word = max(token_len // len(words), 1)
            return '_'.join([w[:len_per_word] for w in words])

        if isinstance(x, str):
            tokens = x.replace("/", "_").split(".")
            tokens = [_initials(t) for t in tokens]
            return ".".join(tokens)
        else:
            return str(x)

    def _abbr(x):
        if isinstance(x, Iterable) and not isinstance(x, str):
            strs = []
            for key in x:
                try:
                    val = x.get(key)
                    strs.append("%s=%s" % (_abbr(key), _abbr(val)))
                except:
                    strs.append("%s" % _abbr(key))
            return "+".join(strs)
        else:
            return _abbr_single(x)

    name = "%04dr%d" % (id, repeat)
    abbr = _abbr(parameters)
    if abbr:
        name += "+" + abbr
    # truncate the entire string if it's beyond the max length
    return name[:max_len]

def learn_dt(input_args):
    # Parameters
    env_task = input_args.env_task
    args = dict(task_definition='custom_reward',
                env_task=env_task,
                max_episode_steps=env_to_time[env_task],
                obv_type='global',
                wall_prob=0.25,
                height=env_to_hw[env_task][0],
                width=env_to_hw[env_task][1],
                incorrect_marker_penalty=True,
                delayed_reward=True,
                seed=random.randint(0, 100000000))
    config = AttrDict()
    config.update(args) 
    env = KarelGymEnv(config)
    env._max_episode_steps = config.max_episode_steps
    env = KarelEnvWrapper(env)
    #env = gym.wrappers.Monitor(env, directory="../karel_vid_dt", force=True, video_callable=lambda episode_id: True)
    custom_args = AttrDict()
    id=input_args.pop("id")
    repeat=input_args.pop("repeat")
    custom_args.update(input_args)
    max_depth = custom_args.max_depth
    n_batch_rollouts = custom_args.n_batch_rollouts
    max_samples = custom_args.max_samples
    max_iters = custom_args.max_iters
    train_frac = custom_args.train_frac
    is_reweight = custom_args.is_reweight
    run_name = _generate_run_name(custom_args, id, repeat)
    #if not os.path.exists(f"../data/karel/ppo/{run_name}"):
    #    os.makedirs(f"../data/karel/ppo/{run_name}")
    #log_fname = f'../data/karel/ppo/{run_name}/karel_dt.log'
    #model_path = f'../data/saved_ppo/karel/{env_task}/saved_conv'
    n_test_rollouts = 10
    save_dirname = f'../data/karel/ppo/{run_name}'
    save_fname = 'dt_policy.pk'
    #save_viz_fname = 'dt_policy.dot'
    student = load_dt_policy(save_dirname, save_fname)

    # Test student
    state_transformer = lambda x: x
    rew = test_policy(env, student, state_transformer, n_test_rollouts)
    print('Final reward: {}'.format(rew), INFO)
    print('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

if __name__ == '__main__':
    #max_depth = [6, 12, 15]
    ##max_depth = [12]
    #n_batch_rollouts = [10]
    #max_samples = [100000, 200000, 400000]
    #is_reweight = [False, True]
    ##grid_search = product(*(environments, max_depth, n_batch_rollouts, max_samples, is_reweight))
    #grid_search = product(*(max_depth, n_batch_rollouts, max_samples, is_reweight))
    #for id, param_config in enumerate(grid_search):
    #    for repeat in range(5):
    #        d, n, s, i = param_config
    d, s, i = 12, 100000, True
    input_args = AttrDict(
        env_task = sys.argv[1],
        max_depth  = d,
        n_batch_rollouts = 10,
        max_samples = s,
        max_iters = 80,
        train_frac = 0.8,
        is_reweight = i,
        id=7,
        repeat=4,
        )
    learn_dt(input_args)
