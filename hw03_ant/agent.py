import random
import numpy as np
import os
import torch
from torch import nn


SEED = 1
np.random.seed(SEED)
random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed(SEED)
torch.backends.cudnn.deterministic = True


class Agent(nn.Module):
    def __init__(self):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(28, 256),
            nn.ELU(),
            nn.Linear(256, 256),
            nn.ELU(),
            nn.Linear(256, 8),
            nn.Tanh()
        )
        # self.model = torch.load(__file__[:-8] + "/agent.pkl")
        self.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.eval()
        
    def act(self, state):
        with torch.no_grad():
            state = torch.tensor(np.array(state)).float()
            action = self.model(state)
            return action.cpu().numpy()
        #return 0 # TODO

    def reset(self):
        pass

