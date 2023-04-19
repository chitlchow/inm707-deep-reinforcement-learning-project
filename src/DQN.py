import torch
from torch import nn

class DQN_Agent():
    def __init__(self, input_dim):
        self.network = DQNetwork(
            input_dim=input_dim,
            output_dim=output_dim
        )
    def

class DQNetwork(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQNetwork, self).__init__()
        self.linear1 = nn.Linear(input_dim, input_dim)
        self.linear2 = nn.Linear(input_dim, output_dim)
        self.softmax = nn.Softmax(dim=4)
    def foward(self, x):
        out = self.linear1(x)
        out = self.linear2(out)
        out = self.softmax(out)
        return out