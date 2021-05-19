import torch
import pickle
import random
from itertools import count
import torch.nn as nn
from torch.distributions import MultivariateNormal
from torch.distributions import Categorical
import numpy as np
from ..util.log import *



################################## set device ##################################

print("============================================================================================")


# set device to cpu or cuda
device = torch.device('cpu')

if(torch.cuda.is_available()): 
    device = torch.device('cuda:0') 
    torch.cuda.empty_cache()
    print("Device set to : " + str(torch.cuda.get_device_name(device)))
else:
    print("Device set to : cpu")
    
print("============================================================================================")




################################## PPO Policy ##################################


class RolloutBuffer:
    def __init__(self):
        self.actions = []
        self.states = []
        self.logprobs = []
        self.rewards = []
        self.is_terminals = []
    

    def clear(self):
        del self.actions[:]
        del self.states[:]
        del self.logprobs[:]
        del self.rewards[:]
        del self.is_terminals[:]


class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, has_continuous_action_space, action_std_init):
        super(ActorCritic, self).__init__()
        self.state_dim = state_dim
        self.has_continuous_action_space = has_continuous_action_space

        if has_continuous_action_space:
            self.action_dim = action_dim
            self.action_var = torch.full((action_dim,), action_std_init * action_std_init).to(device)

        # actor
        if has_continuous_action_space :
            self.actor = nn.Sequential(
                            nn.Linear(state_dim, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Tanh()
                        )
        else:
            q_module_list = [nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3)]
            q_module_list.append(nn.ReLU())
            q_module_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4))
            q_module_list.append(nn.ReLU())
            q_module_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3))
            q_module_list.append(nn.ReLU())
            q_module_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3))
            q_module_list.append(nn.ReLU())
            #q_module_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4))
            #q_module_list.append(nn.ReLU())
            test_conv = nn.Sequential(*q_module_list)
            flat_size = self._infer_flat_size(test_conv)[0]
            self.actor = nn.Sequential(
                            *q_module_list,
                            nn.Flatten(),
                            nn.Linear(flat_size, 64),
                            nn.Tanh(),
                            nn.Linear(64, 64),
                            nn.Tanh(),
                            nn.Linear(64, action_dim),
                            nn.Softmax(dim=-1)
                        )

        
        # critic
        q_module_list = [nn.Conv2d(in_channels=4, out_channels=32, kernel_size=3)]
        q_module_list.append(nn.ReLU())
        q_module_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3))
        q_module_list.append(nn.ReLU())
        q_module_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3))
        q_module_list.append(nn.ReLU())
        q_module_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3))
        q_module_list.append(nn.ReLU())
        q_module_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3))
        q_module_list.append(nn.ReLU())
        #q_module_list.append(nn.Conv2d(in_channels=32, out_channels=32, kernel_size=4))
        #q_module_list.append(nn.ReLU())
        test_conv = nn.Sequential(*q_module_list)
        flat_size = self._infer_flat_size(test_conv)[0]
        self.critic = nn.Sequential(
                        *q_module_list,
                        nn.Flatten(),
                        nn.Linear(flat_size, 64),
                        nn.Tanh(),
                        nn.Linear(64, 64),
                        nn.Tanh(),
                        nn.Linear(64, 1)
                    )

    def _infer_flat_size(self, conv):
        encoder_output = conv(torch.ones(1, *self.state_dim))
        return int(np.prod(encoder_output.size()[1:])), encoder_output.size()[1:]
        
    def set_action_std(self, new_action_std):

        if self.has_continuous_action_space:
            self.action_var = torch.full((self.action_dim,), new_action_std * new_action_std).to(device)
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling ActorCritic::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def forward(self):
        raise NotImplementedError
    

    def act(self, state):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            cov_mat = torch.diag(self.action_var).unsqueeze(dim=0)
            dist = MultivariateNormal(action_mean, cov_mat)
        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)

        action = dist.sample()
        action_logprob = dist.log_prob(action)

        self.log_prob = dist.logits
        
        return action.detach(), action_logprob.detach()
    

    def evaluate(self, state, action):

        if self.has_continuous_action_space:
            action_mean = self.actor(state)
            
            action_var = self.action_var.expand_as(action_mean)
            cov_mat = torch.diag_embed(action_var).to(device)
            dist = MultivariateNormal(action_mean, cov_mat)
            
            # For Single Action Environments.
            if self.action_dim == 1:
                action = action.reshape(-1, self.action_dim)

        else:
            action_probs = self.actor(state)
            dist = Categorical(action_probs)
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_values = self.critic(state)
        
        return action_logprobs, state_values, dist_entropy


class PPO:
    def __init__(self, env, model_path=None, train=False, gamma=0.99, K_epochs=40, eps_clip=0.2, action_std_init=0.6):
        self.num_timesteps=1e6
        self.env = env

        lr_actor = lr_critic = 3e-4

        self.has_continuous_action_space = False

        if self.has_continuous_action_space:
            self.action_std = action_std_init

        self.gamma = gamma
        self.eps_clip = eps_clip
        self.K_epochs = K_epochs
        
        self.buffer = RolloutBuffer()

        state_dim = env.observation_space.shape
        action_dim = env.action_space.n
        self.policy = ActorCritic(state_dim, action_dim, self.has_continuous_action_space, action_std_init).to(device)
        self.optimizer = torch.optim.Adam([
                        {'params': self.policy.actor.parameters(), 'lr': lr_actor},
                        {'params': self.policy.critic.parameters(), 'lr': lr_critic}
                    ])

        self.policy_old = ActorCritic(state_dim, action_dim, self.has_continuous_action_space, action_std_init).to(device)
        self.policy_old.load_state_dict(self.policy.state_dict())
        
        self.MseLoss = nn.MSELoss()
        if model_path and not train:
            self.load(model_path)

        self.model_path = model_path


    def set_action_std(self, new_action_std):
        
        if self.has_continuous_action_space:
            self.action_std = new_action_std
            self.policy.set_action_std(new_action_std)
            self.policy_old.set_action_std(new_action_std)
        
        else:
            print("--------------------------------------------------------------------------------------------")
            print("WARNING : Calling PPO::set_action_std() on discrete action space policy")
            print("--------------------------------------------------------------------------------------------")


    def decay_action_std(self, action_std_decay_rate, min_action_std):
        print("--------------------------------------------------------------------------------------------")

        if self.has_continuous_action_space:
            self.action_std = self.action_std - action_std_decay_rate
            self.action_std = round(self.action_std, 4)
            if (self.action_std <= min_action_std):
                self.action_std = min_action_std
                print("setting actor output action_std to min_action_std : ", self.action_std)
            else:
                print("setting actor output action_std to : ", self.action_std)
            self.set_action_std(self.action_std)

        else:
            print("WARNING : Calling PPO::decay_action_std() on discrete action space policy")

        print("--------------------------------------------------------------------------------------------")

    def predict_q(self, state):
        with torch.no_grad():
            state = torch.FloatTensor(state).to(device)
            action, action_logprob = self.policy_old.act(state)
        return self.policy_old.log_prob.detach().cpu().numpy()

    def predict(self, state, train=False):

        if self.has_continuous_action_space:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            if train:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)

            return action.detach().cpu().numpy().flatten()

        else:
            with torch.no_grad():
                state = torch.FloatTensor(state).to(device)
                action, action_logprob = self.policy_old.act(state)
            if train:
                self.buffer.states.append(state)
                self.buffer.actions.append(action)
                self.buffer.logprobs.append(action_logprob)
                return action.detach().cpu()
            return action.detach().cpu().numpy()


    def update(self):

        # Monte Carlo estimate of returns
        rewards = []
        discounted_reward = 0
        for reward, is_terminal in zip(reversed(self.buffer.rewards), reversed(self.buffer.is_terminals)):
            if is_terminal:
                discounted_reward = 0
            discounted_reward = reward + (self.gamma * discounted_reward)
            rewards.insert(0, discounted_reward)
            
        # Normalizing the rewards
        rewards = torch.tensor(rewards, dtype=torch.float32).to(device)
        rewards = (rewards - rewards.mean()) / (rewards.std() + 1e-7)

        # convert list to tensor
        old_states = torch.squeeze(torch.stack(self.buffer.states, dim=0)).detach().to(device)
        old_actions = torch.squeeze(torch.stack(self.buffer.actions, dim=0)).detach().to(device)
        old_logprobs = torch.squeeze(torch.stack(self.buffer.logprobs, dim=0)).detach().to(device)

        
        # Optimize policy for K epochs
        for _ in range(self.K_epochs):

            # Evaluating old actions and values
            logprobs, state_values, dist_entropy = self.policy.evaluate(old_states, old_actions)

            # match state_values tensor dimensions with rewards tensor
            state_values = torch.squeeze(state_values)
            
            # Finding the ratio (pi_theta / pi_theta__old)
            ratios = torch.exp(logprobs - old_logprobs.detach())

            # Finding Surrogate Loss
            advantages = rewards - state_values.detach()   
            surr1 = ratios * advantages
            surr2 = torch.clamp(ratios, 1-self.eps_clip, 1+self.eps_clip) * advantages

            # final loss of clipped objective PPO
            loss = -torch.min(surr1, surr2) + 0.5*self.MseLoss(state_values, rewards) - 0.01*dist_entropy
            
            # take gradient step
            self.optimizer.zero_grad()
            loss.mean().backward()
            self.optimizer.step()
            
        # Copy new weights into old policy
        self.policy_old.load_state_dict(self.policy.state_dict())

        # clear buffer
        self.buffer.clear()
    
    
    def save(self, checkpoint_path):
        torch.save(self.policy_old.state_dict(), checkpoint_path)
   

    def load(self, checkpoint_path):
        self.policy_old.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
        self.policy.load_state_dict(torch.load(checkpoint_path, map_location=lambda storage, loc: storage))
    
    def eval(self, num_evals=20):
        avg_reward = 0
        for eval in range(num_evals):
            state = np.expand_dims(self.env.reset(), 0)
            cum_reward = 0
            for t in count():
                action = self.predict(state)
                next_state, reward, done, _ = self.env.step(action.item())
                cum_reward += reward
                
                # Move to the next state
                state = np.expand_dims(next_state, 0)
                
                if done:
                    break
            avg_reward += cum_reward
        return avg_reward / num_evals
    
    def interact(self):
        POLICY_UPDATE = 900
        num_timesteps = 0
        next_eval=True
        while num_timesteps < self.num_timesteps:
            # Initialize the environment and state
            state = np.expand_dims(self.env.reset(), 0)
            cum_reward = 0
            for t in count():
                action = self.predict(state, train=True).view(1, 1)
                next_state, reward, done, _ = self.env.step(action.item())
                self.buffer.rewards.append(reward)
                self.buffer.is_terminals.append(done)
                num_timesteps += 1
                cum_reward += reward

                # Observe new state
                if done:
                    next_state = None

                # Move to the next state
                state = np.expand_dims(next_state, 0)

                # Perform one step of the optimization (on the policy network)
                if num_timesteps % POLICY_UPDATE == 0:
                    self.update()
                if num_timesteps % 10000 == 0:
                    next_eval = True
                if done:
                    break
            if next_eval:
                eval_rew = self.eval()
                log(f"Timesteps: {num_timesteps}, Eval reward: {eval_rew}", INFO)
                next_eval = False
        self.save(self.model_path)
