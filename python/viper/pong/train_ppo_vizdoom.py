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
    def __init__(self, env=None, shape=[48, 64, 3]):
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
        self.accumulated_reward += reward
        #perception_vector = self.env._world.get_perception_vector()
        #return (self.observation(ob.astype(np.float32)), np.array(perception_vector, dtype=np.float32)), float(reward), done, {}
        return self.observation(ob.astype(np.float32)), float(self.accumulated_reward) if done else float(0), done, {}

    def reset(self):
        ob = self.observation(np.array(self.env.reset(), dtype=np.float32))
        self.accumulated_reward = 0
        #perception_vector = np.array(self.env._world.get_perception_vector(), np.float32)
        #return ob, perception_vector
        return ob

    def observation(self, obs):
        #new_frame = cv2.cvtColor(obs, cv2.COLOR_RGB2GRAY)
        #resized_screen = cv2.resize(new_frame, self.shape[1:],
        #                            interpolation=cv2.INTER_AREA)
        resized_screen = cv2.resize(obs, self.shape[1:],
                                    interpolation=cv2.INTER_AREA)
        #new_obs = np.array(resized_screen, dtype=np.uint8).reshape(self.shape)
        new_obs = np.array(resized_screen, dtype=np.uint8).transpose((2, 1, 0))
        new_obs = new_obs / 255.0
        return new_obs

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
                #env_task='survive',
                env_task='preloaded',
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
    name = vizdoom_config_file.split("/")[-1]
    if not os.path.exists(f"../data/saved_ppo/vizdoom/{vizdoom_config_file}"):
        os.makedirs(f"../data/saved_ppo/vizdoom/{vizdoom_config_file}")
    model_path = f'../data/saved_ppo/vizdoom/{vizdoom_config_file}/saved_conv'
    log_fname = f'../data/saved_ppo/vizdoom/{vizdoom_config_file}/train_conv.log'
    set_file(log_fname)
    q_func = PPO(env=env, model_path=model_path, train=True)
    q_func.interact()
    

if __name__ == '__main__':
    import sys
    input_args = AttrDict()
    input_args.vizdoom_config_file = environments[3]
    learn_q(input_args)
