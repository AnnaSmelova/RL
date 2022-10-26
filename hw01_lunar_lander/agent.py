import random
import numpy as np
import os
import torch
from torch import nn
from torch.nn import functional as F


class QNetwork(nn.Module):
    """Архитектура нейронной сети, приближающей Q-function"""

    def __init__(self, state_size, action_size, seed=0):
        """
        state_size (int): Размерность состояний
        action_size (int): Размерность действий
        seed (int): Random seed
        """
        super(QNetwork, self).__init__()
        # self.seed = torch.manual_seed(seed)
        self.fc1 = nn.Linear(state_size, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 512)
        self.bn2 = nn.BatchNorm1d(512)
        self.fc3 = nn.Linear(512, action_size)

    def forward(self, state):
        x = self.fc1(state)
        x = self.bn1(x)
        x = F.relu(x)
        x = self.fc2(x)
        x = self.bn2(x)
        x = F.relu(x)
        return self.fc3(x)


class Agent:
    def __init__(self):
        self.device = torch.device('cpu') # torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = QNetwork(8, 4, 0).to(self.device)
        #self.model = torch.load(__file__[:-8] + "/agent.pkl", map_location=self.device)
        self.model.load_state_dict(torch.load(__file__[:-8] + "/agent.pkl"))
        self.model.eval()

    def act(self, state):
        state = torch.from_numpy(state).float().unsqueeze(0).to(self.device)
        action_values = self.model(state)
        return np.argmax(action_values.cpu().data.numpy())
