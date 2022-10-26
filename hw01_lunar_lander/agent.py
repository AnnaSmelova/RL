import random
import numpy as np
import os
import torch


class Agent:
    def __init__(self):
        self.model = torch.load(__file__[:-8] + "/agent.pkl")
        
    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to("cpu")
        self.model.eval()
        action_values = self.model(state)
        return np.argmax(action_values.cpu().data.numpy())
