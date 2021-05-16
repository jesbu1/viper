import torch.nn as nn
import torch

import numpy as np

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
        #q_module_list = [nn.GRUCell(input_size=self.input_shape, hidden_size=self.hidden_size, bias=True)]
        q_module_list = [nn.Linear(in_features=self.input_shape, out_features=self.hidden_size)]
        q_module_list.append(nn.ReLU())
        for i in range(3):
            q_module_list.append(nn.Linear(in_features=self.hidden_size, out_features=self.hidden_size))
            q_module_list.append(nn.ReLU())
        q_module_list.append(nn.Linear(in_features=self.hidden_size, out_features=self.num_actions))
        self.q = nn.Sequential(*q_module_list)
        if model_path and not train:
            self.load_state_dict(torch.load(model_path))
        self.model_path = model_path
        self.gru_hidden_state = torch.zeros((self.batch_size, self.hidden_size))
        self.replay_memory=[]
        self.optimizer = torch.optim.Adam(self.q.parameters())

    def reset_hidden(self):
        self.gru_hidden_state = torch.zeros((self.batch_size, self.hidden_size))

    def replay(self):
        # choose <s,a,r,s',done> experiences randomly from the memory
        minibatch = np.random.choice(self.replay_memory, self.batch_size, replace=True)
        # create one list containing s, one list containing a, etc
        s_l =      torch.from_numpy(np.array(list(map(lambda x: x['s'], minibatch)))).float()
        a_l =      torch.from_numpy(np.array(list(map(lambda x: x['a'], minibatch)))).float()
        r_l =      torch.from_numpy(np.array(list(map(lambda x: x['r'], minibatch)))).float()
        sprime_l = torch.from_numpy(np.array(list(map(lambda x: x['sprime'], minibatch)))).float()
        done_l   = torch.from_numpy(np.array(list(map(lambda x: x['done'], minibatch))))
        # Find q(s', a') for all possible actions a'. Store in list
        # We'll use the maximum of these values for q-update  
        qvals_sprime_l = self.predict_q(sprime_l)
        # Find q(s,a) for all possible actions a. Store in list
        target_f = self.predict_q(s_l)
        # q-update target
        # For the action we took, use the q-update value  
        # For other actions, use the current nnet predicted value
        for i,(s,a,r,qvals_sprime, done) in enumerate(zip(s_l,a_l,r_l,qvals_sprime_l, done_l)): 
            if not done:  
                target = r + self.gamma * torch.max(qvals_sprime)
            else:         
                target = r
            target_f[i][a] = target
        # Update weights of neural network with fit() 
        # Loss function is 0 for actions we didn't take
        self.train(s_l, target_f)

    def train(self, s_l, target_f):
        self.optimizer.zero_grad()
        loss = torch.mean((s_l - target_f)**2)
        loss.backward()
        self.optimizer.step()

    def interact(self):
        timesteps = 0
        while timesteps < self.num_timesteps:
            s = self.env.reset()
            done=False
            r_sum = 0
            while not done: 
                # Uncomment this to see the agent learning
                # env.render()
                
                # Feedforward pass for current state to get predicted q-values for all actions 
                with torch.no_grad():
                    qvals_s = self.predict_q(torch.from_numpy(s).unsqueeze(0).float())
                qvals_s = qvals_s.numpy()
                # Choose action to be epsilon-greedy
                if np.random.random() < self.epsilon:  
                    a = self.env.action_space.sample()
                else:                             
                    a = np.argmax(qvals_s); 
                # Take step, store results 
                sprime, r, done, info = self.env.step(a)
                timesteps += 1
                r_sum += r 
                # add to memory, respecting memory buffer limit 
                if len(self.replay_memory) > self.mem_max_size:
                    self.replay_memory.pop(0)
                self.replay_memory.append({"s":s,"a":a,"r":r,"sprime":sprime,"done":done})
                # Update state
                s=sprime
                # Train the nnet that approximates q(s,a), using the replay memory
                self.replay()
                # Decrease epsilon until we hit a target threshold 
                if self.epsilon > 0.01:      
                    self.epsilon -= 0.001
            print("Total reward:", r_sum)
        torch.save(self.state_dict(), self.model_path)

    def predict_q(self, obs):
        return self.q(obs)

    def predict(self, obs):
        acts = torch.argmax(self.q(obs), -1)
        return acts