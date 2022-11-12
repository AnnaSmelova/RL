import numpy as np
from torch.distributions import Normal
from torch import nn
import torch
import random

SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


def layer_init(layer, std=np.sqrt(2), bias_const=0.0):
    torch.nn.init.orthogonal_(layer.weight, std)
    torch.nn.init.constant_(layer.bias, bias_const)
    return layer


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            layer_init(nn.Linear(22, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 64)),
            nn.Tanh(),
            layer_init(nn.Linear(64, 6), std=0.01),
        )
        self.log_sigma = nn.Parameter(torch.zeros(6))
        self.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.eval()
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            mu = self.model(state)
            sigma = torch.exp(self.log_sigma)
            distribution = Normal(mu, sigma)
            action = distribution.sample()
            action_transformed = torch.tanh(action)
            return action_transformed

    def reset(self):
        pass

