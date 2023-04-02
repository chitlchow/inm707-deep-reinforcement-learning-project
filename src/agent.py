import numpy as np
from snake import Snake
import torch
from torch import nn

class Agent:
    def __init__(self):
        self.Q_matrix = np.zeros((4, 4))
        self.gamma = 0.7
        self.alpha = 0.2
        # Have to be programmed - based on the direction it is going
        self.action_set = ['up', 'down', 'left', 'right']

    def build_Q(self, state):
        return self.
#   Q-Matrix for the agent
#   R-Matrix: Give rewards based on the city block distance
#   Definition:
#    - based on the state (distance between food and snake, direction)

