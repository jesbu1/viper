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
from vizdoom_env.vizdoom_gym_env_wrapper import VizDoomGymEnv
from ..core.rl import *
from .karel import *
from .dqn import *
from ..core.dt import *
from ..util.log import *
from collections import Iterable
from .ppo_viz import PPO
import random
import gym
from gym.spaces import Box
import collections
import cv2
from .train_ppo_vizdoom import StackFrames, VizdoomEnvWrapper

environments = [
                'vizdoom_env/asset/default.cfg',
                'vizdoom_env/asset/scenarios/defend_the_center.cfg',
                'vizdoom_env/asset/scenarios/deadly_corridor.cfg',
                'vizdoom_env/asset/scenarios/defend_the_line.cfg',
                ]

class AttrDict(dict):
    def __init__(self, *args, **kwargs):
        super(AttrDict, self).__init__(*args, **kwargs)
        self.__dict__ = self

def learn_q(input_args):
    # Parameters
    vizdoom_config_file = input_args.vizdoom_config_file
    args = dict(task_definition='custom_reward',
                env_task='survive',
                vizdoom_config_file=vizdoom_config_file,
                max_episode_steps=200,
                obv_type='global',
                delayed_reward=False,
                seed=random.randint(0, 100000000))
    config = AttrDict()
    config.update(args)
    env = VizDoomGymEnv(config)
    env._max_episode_steps = config.max_episode_steps
    env = VizdoomEnvWrapper(env)
    env = StackFrames(env)
    env = gym.wrappers.Monitor(env, directory="../vizdoom_vid_ppo", force=True, video_callable=lambda episode_id: True)
    name = vizdoom_config_file.split("/")[-1]
    if not os.path.exists(f"../data/saved_ppo/vizdoom/{vizdoom_config_file}"):
        os.makedirs(f"../data/saved_ppo/vizdoom/{vizdoom_config_file}")
    model_path = f'../data/saved_ppo/vizdoom/{vizdoom_config_file}/saved_conv'
    log_fname = f'../data/saved_ppo/vizdoom/{vizdoom_config_file}/train_conv.log'
    set_file(log_fname)
    q_func = PPO(env=env, model_path=model_path, train=False)
    q_func.eval(render=True)
    env.close() 

if __name__ == '__main__':
    import sys
    input_args = AttrDict()
    input_args.vizdoom_config_file = environments[2]
    learn_q(input_args)
