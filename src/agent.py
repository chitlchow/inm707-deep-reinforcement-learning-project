import numpy as np
from snake import Snake
import torch
from torch import nn

class Agent:
    def __init__(self):
        self.Q_matrix = np.zeros((4, 4))
        self.gamma = 0.7
        self.alpha = 0.2
        self.action_set = ['up', 'down', 'left', 'right']
