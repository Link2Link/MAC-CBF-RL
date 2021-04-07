
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions import MultivariateNormal, Normal

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

class ActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(ActorCritic, self).__init__()
        # action mean range -1 to 1
        self.actor =  nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, action_dim),
                nn.Tanh()
                )
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        
    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory=None):
        action_mean = self.actor(state)
        cov_mat = torch.diag(self.action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            
        return action.detach()
    
    def evaluate(self, state, action):   
        action_mean = self.actor(state)
        
        action_var = self.action_var.expand_as(action_mean)
        cov_mat = torch.diag_embed(action_var).to(device)
        
        dist = MultivariateNormal(action_mean, cov_mat)
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy


class GaussianActorCritic(nn.Module):
    def __init__(self, state_dim, action_dim, action_std):
        super(GaussianActorCritic, self).__init__()
        # action mean range -1 to 1

        self.fc1 = nn.Linear(state_dim, 64)
        self.fc2 = nn.Linear(64, 32)
        self.fc_mean = nn.Linear(32, action_dim)
        self.fc_log_vari = nn.Linear(32, action_dim)
        
        # critic
        self.critic = nn.Sequential(
                nn.Linear(state_dim, 64),
                nn.Tanh(),
                nn.Linear(64, 32),
                nn.Tanh(),
                nn.Linear(32, 1)
                )
        self.action_var = torch.full((action_dim,), action_std*action_std).to(device)
        

    def forward(self):
        raise NotImplementedError
    
    def act(self, state, memory=None):
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        action_mean = self.fc_mean(x)
        action_var = self.fc_log_vari(x).exp()

        dist = MultivariateNormal(action_mean, torch.diag_embed(action_var))

        action = dist.sample()
        action_logprob = dist.log_prob(action)
        
        if memory is not None:
            memory.states.append(state)
            memory.actions.append(action)
            memory.logprobs.append(action_logprob)
            
        return action.detach()
    
    def evaluate(self, state, action):   
        x = F.tanh(self.fc1(state))
        x = F.tanh(self.fc2(x))
        action_mean = self.fc_mean(x)
        action_var = self.fc_log_vari(x).exp()

        dist = MultivariateNormal(action_mean, torch.diag_embed(action_var))
        
        action_logprobs = dist.log_prob(action)
        dist_entropy = dist.entropy()
        state_value = self.critic(state)
        
        return action_logprobs, torch.squeeze(state_value), dist_entropy