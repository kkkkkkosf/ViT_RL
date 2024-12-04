import torch.nn as nn
import random
from collections import namedtuple
import numpy as np

import torch

Trajectory = namedtuple('Trajectory', ('state', 'action', 'value'))

def init_weight(m):
    if type(m) == nn.Linear:
        size = m.weight.size()
        fan_out = size[0]
        fan_in = size[1]
        variance = np.sqrt(2.0 / (fan_in + fan_out))
        m.weight.data.normal_(0.0, variance)
        m.bias.data.fill_(0.0)

class Generator(nn.Module):

    def __init__(self, dim_state=256, dim_hidden1=512, dim_hidden2=512, dim_action=256, activation=nn.Tanh):
        super(Generator, self).__init__()
        self.policy = nn.Sequential(
            nn.Linear(dim_state, dim_hidden1),
            activation(),
            nn.Linear(dim_hidden1, dim_hidden2),
            activation(),
            nn.Linear(dim_hidden2, dim_hidden1),
            activation(),
            nn.Linear(dim_hidden1, dim_action),
            nn.Tanh()
        )
        self.policy.apply(init_weight)

    def forward(self, x):
        change = self.policy(x)
        return change

    def get_action(self, state, soft_value):
        action = self.forward(state)
        current_action = action
        return state + soft_value * current_action

    def get_log_prob(self, state, soft_value):
        action = self.get_action(state, soft_value)
        action = torch.clamp(action,min = 0.0001)
        action_log = -torch.log(action + 1)
        return action_log

    def get_action_use(self, state):
        action = self.forward(state)
        return action

class CriticModel(nn.Module):

    def __init__(self, dim_state_action=256, dim_hidden=64, dim_out=1, activation=nn.LeakyReLU):
        super(CriticModel, self).__init__()
        self.model = nn.Sequential(
            nn.Linear(dim_state_action, dim_hidden),
            activation(),
            nn.Linear(dim_hidden, dim_out)
        )

        self.model.apply(init_weight)

    def forward(self, x):
        value = self.model(x)
        return value

class Memory(object):

    def __init__(self):
        self.memory = []

    def push(self, *args):
        """Saves a (state, action) pair."""
        self.memory.append(Trajectory(*args))

    def sample(self, batch_size=None):
        if batch_size is None:
            return Trajectory(*zip(*self.memory))
        else:
            random_batch = random.sample(self.memory, batch_size)
            return Trajectory(*zip(*random_batch))

    def append(self, new_memory):
        self.memory += new_memory.memory

    def clear(self):
        self.memory.clear()

    def __len__(self):
        return len(self.memory)
