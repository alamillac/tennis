import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1. / np.sqrt(fan_in)
    return (-lim, lim)

class Actor(nn.Module):
    """Actor (Policy) Model."""

    def __init__(self, state_size, action_bounds, fc1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_bounds (tuple): Min and max values for each action
            fc1_units (int): Number of nodes in first hidden layer
            fc2_units (int): Number of nodes in second hidden layer
        """
        super(Actor, self).__init__()
        env_min, env_max = action_bounds
        action_size = len(env_min)

        self.fc1 = nn.Linear(state_size, fc1_units)
        self.fc2 = nn.Linear(fc1_units, fc2_units)
        self.fc3 = nn.Linear(fc2_units, action_size)
        self.relu = nn.LeakyReLU(0.01)

        self.reset_parameters()

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)

        self.env_min = torch.tensor(env_min, device=self.device, dtype=torch.float32)
        self.env_max = torch.tensor(env_max, device=self.device, dtype=torch.float32)

    def reset_parameters(self):
        self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        self.fc3.weight.data.uniform_(-3e-3, 3e-3)

    def forward(self, state):
        """Build an actor (policy) network that maps states -> actions."""
        state = self._format(state)
        x = self.relu(self.fc1(state))
        x = self.relu(self.fc2(x))
        x = F.tanh(self.fc3(x))
        return self._rescale_action(x)

    def _rescale_action(self, action):
        return self.env_min + (action + 1.0) * 0.5 * (self.env_max - self.env_min)

    def _format(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, device=self.device, dtype=torch.float32)


class Critic(nn.Module):
    """Critic (Value) Model."""

    def __init__(self, state_size, action_size, fc1_units=24, fc2_units=48):
        """Initialize parameters and build model.
        Params
        ======
            state_size (int): Dimension of each state
            action_size (int): Dimension of each action
            fc1_units (int): Number of nodes in the first hidden layer
            fc2_units (int): Number of nodes in the second hidden layer
        """
        super(Critic, self).__init__()
        self.layer_a = nn.Sequential(
            nn.Linear(state_size + action_size, fc1_units),
            nn.LeakyReLU(0.01),
            nn.Linear(fc1_units, fc2_units),
            nn.LeakyReLU(0.01),
            nn.Linear(fc2_units, 1),
        )
        self.layer_b = nn.Sequential(
            nn.Linear(state_size + action_size, fc1_units),
            nn.LeakyReLU(0.01),
            nn.Linear(fc1_units, fc2_units),
            nn.LeakyReLU(0.01),
            nn.Linear(fc2_units, 1),
        )

        device = "cpu"
        if torch.cuda.is_available():
            device = "cuda:0"
        self.device = torch.device(device)
        self.to(self.device)


    def forward(self, state, action):
        """Build a critic (value) network that maps (state, action) pairs -> Q-values."""
        state = self._format(state)
        action = self._format(action)
        x = torch.cat((state, action), dim=1)
        xa = self.layer_a(x)
        xb = self.layer_b(x)
        return xa, xb

    def Qa(self, state, action):
        state = self._format(state)
        action = self._format(action)
        x = torch.cat((state, action), dim=1)
        return self.layer_a(x)

    def _format(self, x):
        if isinstance(x, torch.Tensor):
            return x
        return torch.tensor(x, device=self.device, dtype=torch.float32)
