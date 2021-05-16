from collections import deque, namedtuple
import random
import torch.nn as nn
from itertools import count
import torch

import numpy as np
device = 'cpu'

Transition = namedtuple('Transition',
                        ('state', 'action', 'next_state', 'reward'))


class ReplayMemory(object):

    def __init__(self, capacity):
        self.memory = deque([],maxlen=capacity)

    def push(self, *args):
        """Save a transition"""
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class DQN(nn.Module):
    def __init__(self, env, model_path=None, train=False):
        super(DQN, self).__init__()
        self.env = env
        self.model_path = model_path
        self.num_actions = env.action_space.n
        self.input_shape = env.observation_space.shape[0]
        self.batch_size=32
        self.epsilon=0.1
        self.num_timesteps=1e6
        self.gamma=0.99
        self.hidden_size=100
        self.mem_max_size=100000
        self.q = self.build_network()
        self.q_target = self.build_network()
        if model_path and not train:
            self.load_state_dict(torch.load(model_path))
        self.model_path = model_path
        self.gru_hidden_state = torch.zeros((self.batch_size, self.hidden_size))
        self.memory = ReplayMemory(capacity=self.mem_max_size)
        self.optimizer = torch.optim.Adam(self.q.parameters())

    def build_network(self):
        #q_module_list = [nn.GRUCell(input_size=self.input_shape, hidden_size=self.hidden_size, bias=True)]
        q_module_list = [nn.Linear(in_features=self.input_shape, out_features=self.hidden_size)]
        q_module_list.append(nn.ReLU())
        for i in range(3):
            q_module_list.append(nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
            q_module_list.append(nn.ReLU())
        q_module_list.append(nn.Linear(in_features=self.hidden_size, out_features=self.num_actions))
        return nn.Sequential(*q_module_list)

    def optimize_model(self):
        if len(self.memory) < self.batch_size:
            return
        transitions = self.memory.sample(self.batch_size)
        # Transpose the batch (see https://stackoverflow.com/a/19343/3343043 for
        # detailed explanation). This converts batch-array of Transitions
        # to Transition of batch-arrays.
        batch = Transition(*zip(*transitions))

        # Compute a mask of non-final states and concatenate the batch elements
        # (a final state would've been the one after which simulation ended)
        non_final_mask = torch.tensor(tuple(map(lambda s: s is not None,
                                            batch.next_state)), device=device, dtype=torch.bool)
        non_final_next_states = torch.cat([s for s in batch.next_state
                                                    if s is not None])
        state_batch = torch.cat(batch.state)
        action_batch = torch.cat(batch.action)
        reward_batch = torch.cat(batch.reward)

        # Compute Q(s_t, a) - the model computes Q(s_t), then we select the
        # columns of actions taken. These are the actions which would've been taken
        # for each batch state according to policy_net
        state_action_values = self.q(state_batch).gather(1, action_batch)

        # Compute V(s_{t+1}) for all next states.
        # Expected values of actions for non_final_next_states are computed based
        # on the "older" target_net; selecting their best reward with max(1)[0].
        # This is merged based on the mask, such that we'll have either the expected
        # state value or 0 in case the state was final.
        next_state_values = torch.zeros(self.batch_size, device=device)
        next_state_values[non_final_mask] = self.q_target(non_final_next_states).max(1)[0].detach()
        # Compute the expected Q values
        expected_state_action_values = (next_state_values * self.gamma) + reward_batch

        # Compute Huber loss
        criterion = nn.SmoothL1Loss()
        loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))

        # Optimize the model
        self.optimizer.zero_grad()
        loss.backward()
        for param in self.q.parameters():
            param.grad.data.clamp_(-1, 1)
        self.optimizer.step()

    def reset_hidden(self):
        self.gru_hidden_state = torch.zeros((self.batch_size, self.hidden_size))

    def interact(self):
        TARGET_UPDATE = 1000
        num_timesteps = 0
        while num_timesteps < self.num_timesteps:
            # Initialize the environment and state
            state = torch.from_numpy(self.env.reset())
            for t in count():
                # Select and perform an action
                if np.random.random() < self.epsilon:  
                    action = self.env.action_space.sample()
                else:                             
                    action = self.predict(state.unsqueeze(0))[0].item()
                next_state, reward, done, _ = self.env.step(action.item())
                next_state = torch.from_numpy(next_state, device=device)
                reward = torch.tensor([reward], device=device)

                # Observe new state
                if done:
                    next_state = None

                # Store the transition in memory
                self.memory.push(state, action, next_state, reward)

                # Move to the next state
                state = next_state

                # Perform one step of the optimization (on the policy network)
                self.optimize_model()
                if self.epsilon > 0.01:      
                    self.epsilon -= 0.001
                if done:
                    break
            # Update the target network, copying all weights and biases in DQN
            if num_timesteps % TARGET_UPDATE == 0:
                self.q_target.load_state_dict(self.q.state_dict())

        torch.save(self.state_dict(), self.model_path)

    def predict_q(self, obs):
        return self.q(obs)

    def predict(self, obs):
        acts = torch.argmax(self.q(obs), -1)
        return acts