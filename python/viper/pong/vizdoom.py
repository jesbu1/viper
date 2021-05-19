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

from vizdoom_env.vizdoom_gym_env_wrapper import VizDoomGymEnv
from ..core.rl import *
from .karel import *
from ..core.dt import *
from ..util.log import *
#from .custom_dqn import DQN
from .ppo_viz import PPO
from collections import Iterable
import random
from itertools import product
import gym
from gym.spaces import Box
import sys
import cv2
import collections

class StackFrames(gym.ObservationWrapper):
  #init the new obs space (gym.spaces.Box) low & high bounds as repeat of n_steps. These should have been defined for vizdooom
  
  #Create a return a stack of observations
    def __init__(self, env, repeat=2):
        super(StackFrames, self).__init__(env)
        self.observation_space = gym.spaces.Box( env.observation_space.low.repeat(repeat, axis=0),
                              env.observation_space.high.repeat(repeat, axis=0),
                            dtype=np.float32)
        self.stack = collections.deque(maxlen=repeat)
    def reset(self):
        self.stack.clear()
        observation = self.env.reset()
        for _ in range(self.stack.maxlen):
            self.stack.append(observation)
        return  np.array(self.stack).reshape(self.observation_space.low.shape)
    def observation(self, observation):
        self.stack.append(observation)
        return np.array(self.stack).reshape(self.observation_space.low.shape)

class VizdoomEnvWrapper(gym.Wrapper):
    def __init__(self, env=None, shape=[64, 48, 1]):
        """
        Transpose observation space for images
        """
        gym.Wrapper.__init__(self, env)
        obs_shape = self.observation_space.shape
        self.shape = (shape[2], shape[0], shape[1])
        if len(obs_shape) == 3:
            self.observation_space = gym.spaces.Box(low=0.0, high=1.0,
                                        shape=self.shape, dtype=np.float32)

    def step(self, action):
        ob, reward, done, info = self.env.step(action)
        #perception_vector = self.env._world.get_perception_vector()
        #return (self.observation(ob.astype(np.float32)), np.array(perception_vector, dtype=np.float32)), float(reward), done, {}
        return self.observation(ob.astype(np.float32)), float(reward), done, {}

    def reset(self):
        ob = self.observation(np.array(self.env.reset(), dtype=np.float32))
        #perception_vector = np.array(self.env._world.get_perception_vector(), np.float32)
        #return ob, perception_vector
        return ob

    def observation(self, obs):
        new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        resized_screen = cv2.resize(new_frame, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = new_obs / 255.0
        return new_obs

environments = [
                'vizdoom_env/asset/default.cfg',
                ]
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
    vizdoom_config_file = input_args.vizdoom_config_file
    args = dict(task_definition='custom_reward',
                env_task='survive',
                #vizdoom_config_file='vizdoom_env/asset/default.cfg',
                vizdoom_config_file=vizdoom_config_file,
                max_episode_steps=100,
                obv_type='global',
                delayed_reward=False,
                seed=random.randint(0, 100000000))
    config = AttrDict()
    config.update(args) 
    env = VizDoomGymEnv(config)
    env._max_episode_steps = config.max_episode_steps
    env = VizdoomEnvWrapper(env)
    env = StackFrames(env)
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
    if not os.path.exists(f"../data/vizdoom/ppo/{run_name}"):
        os.makedirs(f"../data/vizdoom/ppo/{run_name}")
    log_fname = f'../data/vizdoom/ppo/{run_name}/karel_dt.log'
    #model_path = f'../data/saved_dqn/karel/{env_task}/saved'
    model_path = f'../data/saved_ppo/vizdoom/{vizdoom_config_file.split("/")[-1]}/saved_conv'
    n_test_rollouts = 50
    save_dirname = f'../data/vizdoom/ppo/{run_name}'
    save_fname = 'dt_policy.pk'
    save_viz_fname = 'dt_policy.dot'
    is_train = True
    
    # Logging
    set_file(log_fname)
    
    # Data structures
    teacher = PPO(env, model_path, train=False)
    student = DTPolicy(max_depth)
    state_transformer = lambda x: x

    # Train student
    if is_train:
        student = train_dagger(env, teacher, student, state_transformer, max_iters, n_batch_rollouts, max_samples, train_frac, is_reweight, n_test_rollouts)
        save_dt_policy(student, save_dirname, save_fname)
        save_dt_policy_viz(student, save_dirname, save_viz_fname)
    else:
        student = load_dt_policy(save_dirname, save_fname)

    # Test student
    rew = test_policy(env, student, state_transformer, n_test_rollouts)
    log('Final reward: {}'.format(rew), INFO)
    log('Number of nodes: {}'.format(student.tree.tree_.node_count), INFO)

if __name__ == '__main__':
    max_depth = [6, 12, 15]
    #max_depth = [12]
    n_batch_rollouts = [10]
    max_samples = [100000, 200000, 400000]
    is_reweight = [False, True]
    #grid_search = product(*(environments, max_depth, n_batch_rollouts, max_samples, is_reweight))
    grid_search = product(*(max_depth, n_batch_rollouts, max_samples, is_reweight))
    for id, param_config in enumerate(grid_search):
        for repeat in range(5):
            d, n, s, i = param_config
            input_args = AttrDict(
                vizdoom_config_file = sys.argv[1],
                max_depth  = d,
                n_batch_rollouts = n,
                max_samples = s,
                max_iters = 80,
                train_frac = 0.8,
                is_reweight = i,
                id=id,
                repeat=repeat,
                )
            learn_dt(input_args)
