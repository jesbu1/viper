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

from os import environ
from karel_env.karel_gym_env import KarelGymEnv
from ..core.rl import *
from .karel import *
from .dqn import *
from ..core.dt import *
from ..util.log import *
from collections import Iterable
from .custom_dqn import DQN
import random
import gym
from gym.spaces import Box

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
        return self.observation(ob.astype(np.float32)), float(reward), done, {}

    def reset(self):
        ob = self.observation(np.array(self.env.reset(), dtype=np.float32))
        return ob

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
    randomMaze=(8, 8),
    stairClimber_sparse=(12, 12),
    topOff=(12, 12),
)
env_to_time = dict(
    cleanHouse=300,
    fourCorners=100,
    harvester=100,
    randomMaze=100,
    stairClimber_sparse=100,
    topOff=100,
)

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def learn_q(input_args):
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
    if not os.path.exists(f"../data/saved_dqn/karel/{env_task}"):
        os.makedirs(f"../data/saved_dqn/karel/{env_task}")
    model_path = f'../data/saved_dqn/karel/{env_task}/saved_conv'
    q_func = DQN(env=env, model_path=model_path, train=True, conv=True)
    q_func.interact()
    

if __name__ == '__main__':
    for env in environments:
        input_args = AttrDict()
        input_args.env_task = env
        learn_q(input_args)